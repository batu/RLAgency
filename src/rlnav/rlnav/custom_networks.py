from typing import Any, Dict, List, Optional, Tuple, Type

import gym
import torch as th
from torch._C import device
import torch.nn.functional as F
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy, create_sde_features_extractor, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from torch import nn
import numpy as np

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class TransformerNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        squash_output: bool = False, ):
        super(TransformerNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.squash_output = squash_output
        self.hidden_dim = hidden_dim
        self.action_dim = 3 
         
        self.isCritic = output_dim > 0 
        self.model = self.create_model()

    def create_model(self):

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=1, dim_feedforward=128, dropout=0)
        output_size = self.output_dim if self.isCritic else self.hidden_dim
        model_list =[
            nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, output_size), nn.ReLU(),
        ]
        if self.squash_output:
            model_list.append(nn.Tanh())

        return nn.Sequential(*model_list)

    def forward(self, input_tensor: th.Tensor) -> th.Tensor:
        
        input_tensor = th.unsqueeze(input_tensor, 1)
        encoded = self.transformer_layer(input_tensor).squeeze()
        output = self.model(encoded)
        return output

class MyGlobal3DMaxPoolLayer(nn.Module): #Is this the correct name?
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return th.mean(x, dim=(-3, -2,-1))

class MyGlobal2DMaxPoolLayer(nn.Module): #Is this the correct name?
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return th.mean(x, dim =(-2,-1))


class CustomNetwork3D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        squash_output: bool = False, ):
        super(CustomNetwork3D, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.squash_output = squash_output
        self.hidden_dim = hidden_dim

        self.action_dim = 3 

        vector_size = 19 + 26 + 6  # This includes the whiskers
        depth_size = 55 - 6
        occupancy_size = 405

        self.vector_end = vector_size
        self.depthmask_end = self.vector_end + depth_size
        self.occupancy_end = self.depthmask_end + occupancy_size
 
        self.depthmask_size = 7
        self.occupancy_shape = (9, 5, 9)

        self.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
 
        self.is_critic = output_dim > 0 
        self.dummy_input = th.rand((512, self.input_dim + self.action_dim)).cuda() if self.is_critic else th.rand((512, self.input_dim)).cuda()
        
        self.create_model()
        # self.simple_model = self.create_simple_model()

    def create_simple_model(self):
        output_size = self.output_dim if self.is_critic else self.hidden_dim
        model_list =[
            nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim , self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, output_size), nn.ReLU(),
        ]

        if self.squash_output:
            model_list.append(nn.Tanh())

        return nn.Sequential(*model_list)

    def create_model(self):
        _, dummy_depth_obs2d, dummy_occupancy_obs3d = self.preprocess_2dnp(self.dummy_input)

        def create_depthmap_layers(): 
            conv2d_size = 32
            depthmap_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=conv2d_size, kernel_size=3, padding=2, stride=1), nn.ReLU(),
                nn.Conv2d(conv2d_size,   out_channels=conv2d_size, kernel_size=3, padding=2, stride=1), nn.ReLU(),
                MyGlobal2DMaxPoolLayer(),
            ) 
            return depthmap_layers.to(self.device)

        def create_occupancy2D_layers():
            conv2d_size = 32
            occupancy_layers = nn.Sequential(
                nn.Conv2d(in_channels=9, out_channels=conv2d_size, kernel_size=3, padding=2, stride=1), nn.ReLU(),
                nn.Conv2d(conv2d_size,   out_channels=conv2d_size, kernel_size=3, padding=2, stride=1), nn.ReLU(),
                MyGlobal2DMaxPoolLayer()
            )
            return occupancy_layers.to(self.device)

        def create_occupancy_layers():
            conv3d_size = 32
            occupancy_layers = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=conv3d_size, kernel_size=3, padding=2, stride=1), nn.ReLU(),
                nn.Conv3d(conv3d_size,   out_channels=conv3d_size, kernel_size=3, padding=2, stride=1), nn.ReLU(),
                MyGlobal3DMaxPoolLayer()
            )
            return occupancy_layers.to(self.device)

        def create_vector_layers():
            # If you are the critic your vector obs also involes the actions from the actor
            vector_input_size = self.vector_end + self.action_dim if self.is_critic else self.vector_end
            vector_layers = nn.Sequential(
                nn.Linear(vector_input_size, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim), 
            )
            return vector_layers.to(self.device)

        self.depthmask_layers = create_depthmap_layers()
        depthmask_output_shape = self.depthmask_layers(dummy_depth_obs2d).shape[-1]

        self.occupancy_layers = create_occupancy2D_layers()
        occupancy_output_shape = self.occupancy_layers(dummy_occupancy_obs3d).shape[-1]

        self.vector_layers = create_vector_layers()

        # Whether the output 3 actions, or the hidden size (the mu and sigma used for SAC actor is created later, outside the scope of this class)
        output_size = self.output_dim if self.is_critic else self.hidden_dim
        model_list =[
            nn.Linear(self.hidden_dim + occupancy_output_shape + depthmask_output_shape, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, output_size), nn.ReLU(),
        ]

        if self.squash_output:
            model_list.append(nn.Tanh())

        # th.backends.cudnn.benchmark = True
        self.combo_model = nn.Sequential(*model_list)

    def dict_preprocess(self, input_dict):
        vector_obs = input_dict["vector"]
        depth_obs2d = input_dict["depthmap"]
        occupancy_obs3d = input_dict["occupancy"]  

        return vector_obs, depth_obs2d, occupancy_obs3d

    def preprocess_2dnp(self, input_list):

        input_array = th.squeeze(input_list).cuda()
        vector_obs = input_array[:, :self.vector_end]
        depth_obs = input_array[:,  self.vector_end:self.depthmask_end]
        occupancy_obs = input_array[:, self.depthmask_end: self.occupancy_end]     

        depth_obs2d = depth_obs.reshape(-1, 1, 7, 7)
        occupancy_obs2d = occupancy_obs.reshape(-1,  9, 5, 9)

        return vector_obs, depth_obs2d, occupancy_obs2d


    def preprocess(self, input_tensor):
        vector_obs = th.cat((input_tensor[:, :self.vector_end], input_tensor[:, -self.action_dim:]), dim=1) if self.is_critic else input_tensor[:, :self.vector_end]
        depth_obs = input_tensor[:, self.vector_end:self.depthmask_end]
        occupancy_obs = input_tensor[:, self.depthmask_end: self.occupancy_end]     

        depth_obs2d = depth_obs.reshape(-1, 1, self.depthmask_size, self.depthmask_size)
        occupancy_obs3d = occupancy_obs.reshape(-1, 1, *self.occupancy_shape)

        return vector_obs, depth_obs2d, occupancy_obs3d

    def preprocess_np(self, input_tensor):
        input_array = input_tensor.detach().cpu().numpy()

        vector_obs = np.concatenate((input_array[:, :self.vector_end], input_array[:, -self.action_dim:]), axis=1) if self.is_critic else input_array[:, :self.vector_end]
        depth_obs = input_array[:, self.vector_end:self.depthmask_end]
        occupancy_obs = input_array[:, self.depthmask_end: self.occupancy_end]     

        depth_obs2d = depth_obs.reshape(-1, 1, self.depthmask_size, self.depthmask_size)
        occupancy_obs3d = occupancy_obs.reshape(-1, 1, *self.occupancy_shape)

        return th.tensor(vector_obs).cuda(), th.tensor(depth_obs2d).cuda(), th.tensor(occupancy_obs3d).cuda()


    def forward(self, input_tensor: th.Tensor) -> th.Tensor:
        # Divide the incoming 1d observation into three parts, reshape it as necesarry.
        # if it is the critic, append the actions (given as input) to the end of the vector obs
        vector_obs, depth_obs2d, occupancy_obs3d = self.dict_preprocess(input_tensor)
        
        # Reshape to the correct size and pass it through the network.
        vector_output = self.vector_layers(vector_obs)
        depth_output = th.squeeze(self.depthmask_layers(depth_obs2d))
        occupancy_output = th.squeeze(self.occupancy_layers(occupancy_obs3d))
        
        # Get the combination and pass it through the last linear layers.
        combined_input = th.cat((vector_output, occupancy_output, depth_output), dim=1)
        output = self.combo_model(combined_input)
        # output = self.simple_model(input_tensor)
        return output


class CustomActor(BasePolicy):
    """
    Actor network (policy) for SAC.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        # self.latent_pi = TransformerNetwork(features_dim, -1, net_arch[0])
        self.latent_pi = CustomNetwork3D(features_dim, -1, net_arch[0])
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            latent_sde_dim = last_layer_dim
            # Separate features extractor for gSDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                    features_dim, sde_net_arch, activation_fn
                )

            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=latent_sde_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                sde_net_arch=self.sde_net_arch,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).
        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.
        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            latent_sde = latent_pi
            if self.sde_features_extractor is not None:
                latent_sde = self.sde_features_extractor(features)
            return mean_actions, self.log_std, dict(latent_sde=latent_sde)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)


class CustomContinuousCritic(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = TransformerNetwork(features_dim + action_dim, 1, net_arch[0])
            q_net = CustomNetwork3D(features_dim + action_dim, 1, net_arch[0])
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        
        if isinstance(features, dict):
            features["vector"] = th.cat([features["vector"], actions], dim=1)
        else:
            features = th.cat([features, actions], dim=1)
        return tuple(q_net(features) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class SACCustomPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(self, *args, **kwargs):
        super(SACCustomPolicy, self).__init__(
            *args,
            **kwargs,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)


register_policy("SACCustomPolicy", SACCustomPolicy)

from typing import Any, Dict, List, Optional, Tuple, Type

import gym
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy, create_sde_features_extractor, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from torch import nn

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class DenseMlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        squash_output: bool = False,
    ):
        super(DenseMlp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.squash_output = squash_output
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim + input_dim, hidden_dim)
        if output_dim > 0:
            self.output_layer = nn.Linear(hidden_dim + input_dim, output_dim)
        self.output_activation = nn.Tanh()

    def forward(self, input_tensor: th.Tensor) -> th.Tensor:
        out_1 = F.relu(self.layer_1(input_tensor))
        input_2 = th.cat([out_1, input_tensor], dim=1)
        out_2 = F.relu(self.layer_2(input_2))

        if self.output_dim < 0:
            return out_2

        input_3 = th.cat([out_2, input_tensor], dim=1)
        out_3 = self.output_layer(input_3)

        if self.squash_output:
            out_3 = self.output_activation(out_3)
        return out_3


class DenseActor(BasePolicy):
    """
    Actor network (policy) for SAC.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        self.latent_pi = DenseMlp(features_dim, -1, net_arch[0])
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            latent_sde_dim = last_layer_dim
            # Separate features extractor for gSDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                    features_dim, sde_net_arch, activation_fn
                )

            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=latent_sde_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                sde_net_arch=self.sde_net_arch,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).
        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.
        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            latent_sde = latent_pi
            if self.sde_features_extractor is not None:
                latent_sde = self.sde_features_extractor(features)
            return mean_actions, self.log_std, dict(latent_sde=latent_sde)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic)


class DenseContinuousCritic(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = DenseMlp(features_dim + action_dim, 1, net_arch[0])
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class SACDensePolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(self, *args, **kwargs):
        super(SACDensePolicy, self).__init__(
            *args,
            **kwargs,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DenseActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DenseActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DenseContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DenseContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic)


register_policy("DenseMlpPolicy", SACDensePolicy)