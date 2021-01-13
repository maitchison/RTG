import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel as DP
import math

import utils
from utils import check_tensor

from marl_env import MultiAgentVecEnv

class BaseEncoder(nn.Module):

    def __init__(self, input_dims, out_features):
        """
        :param input_dims: intended dims of input, excluding batch size (channels, height, width)
        """
        super().__init__()

        self.input_dims = input_dims
        self.final_dims = None
        self.out_features = out_features

    def forward(self, x):
        raise NotImplemented()

class DefaultEncoder(BaseEncoder):
    """ Encodes from observation to hidden representation. """

    def __init__(self, input_dims, out_features=128):
        """
        :param input_dims: intended dims of input, excluding batch size (height, width, channels)
        """
        super().__init__(input_dims, out_features)

        self.final_dims = (64, self.input_dims[1]//2//2//2, self.input_dims[2]//2//2//2)

        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(utils.prod(self.final_dims), self.out_features)

        print(f" -created default encoder, final dims {self.final_dims}")

    def forward(self, x):
        """
        input float32 tensor of dims [b, c, h, w]
        return output tensor of dims [b, d], where d is the number of units in final layer.
        """
        assert type(x) == torch.Tensor, f"Input must be torch tensor not {type(x)}"
        assert x.shape[1:] == self.input_dims, f"Input dims {x.shape[1:]} must match {self.input_dims}"
        assert x.dtype == torch.float32, f"Datatype should be torch.float32 not {x.dtype}"

        b = x.shape[0]

        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        assert x.shape[1:] == self.final_dims, f"Expected final shape to be {self.final_dims} but found {x.shape[1:]}"
        x = x.reshape((b, -1))
        x = torch.sigmoid(self.fc(x))  # try sigmoid, see if it's any better than relu (I'm getting a lot of 0 activations...)

        return x

class FastEncoder(BaseEncoder):
    """ Encodes from observation to hidden representation.
        Optimized to be efficient on CPU
    """

    def __init__(self, input_dims, out_features=128):
        """
        :param input_dims: intended dims of input, excluding batch size (height, width, channels)
        """
        super().__init__(input_dims, out_features)

        self.final_dims = (64, self.input_dims[1]//6, self.input_dims[2]//6)

        # based on nature

        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(utils.prod(self.final_dims), self.out_features)

        print(f" -created fast encoder, final dims {self.final_dims}")

    def forward(self, x):
        """
        input float32 tensor of dims [b, c, h, w]
        return output tensor of dims [b, d], where d is the number of units in final layer.
        """

        assert x.shape[1:] == self.input_dims, f"Input dims {x.shape[1:]} must match {self.input_dims}"
        assert x.dtype == torch.float32, f"Datatype should be torch.float32 not {x.dtype}"

        b = x.shape[0]

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        assert x.shape[1:] == self.final_dims, f"Expected final shape to be {self.final_dims} but found {x.shape[1:]}"
        x = x.reshape((b, -1))
        x = torch.relu(self.fc(x))

        return x

class BaseDecoder(nn.Module):

    def __init__(self, output_dims, initial_dims, hidden_features):
        super().__init__()
        self.output_dims = output_dims  # (c, h, w)
        self.initial_dims = initial_dims
        self.hidden_features = hidden_features

class DefaultDecoder(BaseDecoder):
    """ Decodes from hidden representation to observation. """

    def __init__(self, output_dims, hidden_features=128):
        """
        :param output_dims: (c,h,w)
        :param hidden_features:
        """

        initial_dims = (64, math.ceil((output_dims[1] + 2) / 8), math.ceil((output_dims[2] + 2) / 8))
        super().__init__(output_dims, initial_dims, hidden_features)

        self.fc = nn.Linear(self.hidden_features, utils.prod(self.initial_dims))
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, self.output_dims[0], kernel_size=4, stride=2, padding=1)

        print(f" -created default decoder, initial dims {self.initial_dims}")

    def forward(self, x):
        """
        input tensor of dims [b, d], where d is the number of hidden features.
        return tensor of dims [b, c, h, w]
        """

        b = x.shape[0]

        x = torch.relu(self.fc(x))
        x = x.view((b, *self.initial_dims))
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.clamp(torch.sigmoid(1.2*self.deconv3(x))-0.1, min=0.0, max=1.0) # extend range just make it easier to hit 0 and 1 pixels

        # the decoder doubles the size each time, edges tend not to predict well anyway due to padding
        # so we calculate the required dims (rx,ry) and the excess (ex,ry), then take a center crop
        rx, ry = self.output_dims[-1], self.output_dims[-2]
        ex, ey = x.shape[-1] - rx, x.shape[-2] - ry
        x = x[:, :, ey//2:ey//2+ry, ex//2:ex//2+rx]

        return x


class BaseModel(nn.Module):

    def __init__(
            self,
            env: MultiAgentVecEnv,
            device="cpu",
            dtype=torch.float32,
            memory_units=256,
            out_features=256,
            model="default",
            data_parallel=False,
            lstm_mode='cat',
            nan_check=False,

    ):
        assert env.observation_space.dtype == np.uint8, "Observation space should be 8bit integer"

        self.lstm_mode = lstm_mode
        if self.lstm_mode == 'residual':
            assert memory_units == out_features
            self.encoder_output_features = out_features
        elif self.lstm_mode == 'off':
            self.encoder_output_features = out_features
        elif self.lstm_mode == 'on':
            self.encoder_output_features = memory_units
        elif self.lstm_mode == 'cat':
            self.encoder_output_features = memory_units + out_features
        else:
            raise ValueError(f"invalid lstm mode {self.lstm_mode}.")

        super().__init__()

        # environment is channels last, but we work with channels first.
        self.input_dims = env.observation_space.shape

        self.n_actions = env.action_space.n
        self.device = device
        self.dtype = dtype
        self.lstm_mode = lstm_mode
        self.nan_check = nan_check

        if model.lower() == "default":
            self.encoder = DefaultEncoder(env.observation_space.shape, out_features=out_features)
        elif model.lower() == "fast":
            self.encoder = FastEncoder(env.observation_space.shape, out_features=out_features)
        else:
            raise Exception(f"Invalid model {model}, expected [default|fast|global]")

        self.encoder_features = self.encoder.out_features
        self.memory_units = memory_units

        # note, agents are AI controlled, players maybe scripted, but an AI still needs to predict their behaviour.
        self.n_agents = env.total_agents
        self.n_players = env.max_players

        # hardcode this to 3 for simplicity, could use env.max_roles, but then it might change from game to game
        self.n_roles = 3
        self.obs_shape = env.observation_space.shape

        # memory units
        self.lstm = torch.nn.LSTM(input_size=self.encoder_features, hidden_size=self.memory_units, num_layers=1,
                                  batch_first=False, dropout=0)

        self.encoder_type = type(self.encoder) # the type of encoder before and data_parallel was applied
        if data_parallel:
            # enable multi gpu :)
            print(f" -enabling {utils.Color.OKGREEN}Multi-GPU{utils.Color.ENDC} support")
            self.encoder = DP(self.encoder)

    def forward_sequence(self, obs, rnn_states, terminals=None):
        raise NotImplemented()

    def _check_for_nans(self, tensors_to_check):
        """
        Checks for nans and extreme values in given tensor list
        :param tensors_to_check: list of tuple (name, tensor)
        :return:
        """
        for k, v in tensors_to_check:
            has_nan = torch.any(v.isnan())
            v_min, v_max = torch.min(v), torch.max(v)
            if has_nan:
                raise Exception("nan found in output")
            if abs(v_min) > 1000 or abs(v_max) > 1000:
                print(f"Warning, extreme values found on {k} min:{v_min}, max:{v_max}")

    def _forward_sequence(self, obs, rnn_states, terminals=None):
        """
        Forward a sequence of observations through model, returns dictionary of outputs.
        :param obs: input observations, tensor of dims [N, B, observation_shape], which should be channels last. (BCHW)
        :param rnn_states: float32 tensor of dims [B, 2, memory_dims] containing the initial rnn h,c states
        :param terminals: (optional) tensor of dims [N, B] indicating timesteps that are terminal
        :return: lstm outputs

        """

        N, B, *obs_shape = obs.shape

        # merge first two dims into a batch, run it through encoder, then reshape it back into the correct form.
        assert tuple(obs_shape) == self.obs_shape, f"Expected obs_input to be in the form [N, B, {self.obs_shape}] but found {obs.shape}"
        assert tuple(rnn_states.shape) == (B, 2, self.memory_units), f"Expected {(B, 2, self.memory_units)} found {rnn_states.shape}."
        if terminals is not None:
            assert tuple(terminals.shape) == (N, B)

        if terminals is not None and terminals.sum() == 0:
            # just ignore terminals if there aren't any (faster)
            terminals = None

        obs = obs.reshape([N*B, *obs_shape])
        obs = self.prep_for_model(obs)

        encoding = self.encoder(obs)
        encoding = encoding.reshape([N, B, -1])

        # torch requires these to be contiguous, it's a shame we have to do this but it keeps things simpler to use
        # one tensor rather than a separate one for h and c.
        # also, h,c should be dims [1, B, memory_units] as we have 1 LSTM layer and it is unidirectional
        h = rnn_states[np.newaxis, :, 0, :].clone().detach().contiguous()
        c = rnn_states[np.newaxis, :, 1, :].clone().detach().contiguous()

        if terminals is None:
            # this is the faster path if there are no terminals
            lstm_output, (h, c) = self.lstm(encoding, (h, c))
        else:
            # this is about 2x slower, and takes around 100ms on a batch size of 4096
            outputs = []
            for t in range(N):
                terminals_t = terminals[t][np.newaxis, :, np.newaxis]
                h = h * (1.0 - terminals_t)
                c = c * (1.0 - terminals_t)
                output, (h, c) = self.lstm(encoding[t:t+1], (h, c))
                outputs.append(output)
            lstm_output = torch.cat(outputs, dim=0)

        # copy new rnn states into a new tensor
        new_rnn_states = torch.zeros_like(rnn_states)
        new_rnn_states[:, 0, :] = h[0].detach()
        new_rnn_states[:, 1, :] = c[0].detach()

        if self.lstm_mode == 'residual':
            output = lstm_output + encoding
        elif self.lstm_mode == 'off':
            output = encoding
        elif self.lstm_mode == 'on':
            output = lstm_output
        elif self.lstm_mode == 'cat':
            output = torch.cat([lstm_output, encoding], dim=2)
        else:
            raise ValueError(f"invalid lstm mode {self.lstm_mode}.")

        return output, new_rnn_states

    def prep_for_model(self, x, scale_int=True):
        """ Converts data to format for model (i.e. uploads to GPU, converts type).
            Can accept tensor or ndarray.
            scale_int scales uint8 to [0..1]
         """

        assert self.device is not None, "Must call set_device_and_dtype."

        utils.validate_dims(x, (None, *self.input_dims))

        # if this is numpy convert it over
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)

        # move it to the correct device
        x = x.to(device=self.device, non_blocking=True)

        # then covert the type (faster to upload uint8 then convert on GPU)
        if x.dtype == torch.uint8:
            x = x.to(dtype=self.dtype, non_blocking=True)
            if scale_int:
                x = x / 255
        elif x.dtype == self.dtype:
            pass
        else:
            raise Exception("Invalid dtype {}".format(x.dtype))

        return x

    def set_device_and_dtype(self, device, dtype):

        self.to(device)
        if str(dtype) in ["torch.half", "torch.float16"]:
            self.half()
        elif str(dtype) in  ["torch.float", "torch.float32"]:
            self.float()
        elif str(dtype) in ["torch.double", "torch.float64"]:
            self.double()
        else:
            raise Exception("Invalid dtype {} for model.".format(dtype))

        self.device, self.dtype = device, dtype

    def forward(self, x):
        """
        This is just here so we can get a graph of this module.
        :param x:
        :return:
        """
        fake_memory_states = torch.zeros([x.shape[1], 2, self.memory_units], dtype=torch.float32, device=x.device)
        results, states = self.forward_sequence(x, fake_memory_states)
        return results['role_log_policy']


class DeceptionModel(BaseModel):

    def __init__(
            self,
            env: MultiAgentVecEnv,
            n_predictions,
            device="cpu",
            dtype=torch.float32,
            memory_units=512,
            out_features=512,
            model="default",
            data_parallel=False,
            lstm_mode='residual',
            predict='full',          # off|forward|full
            predict_observations=True,
            predict_actions=True,
            nan_check=False,
    ):
        super().__init__(env, device, dtype, memory_units, out_features, model, data_parallel, lstm_mode, nan_check)


        assert predict in ['off', 'forward', 'full']
        self.predict = predict

        # prediction of each role
        self.role_prediction_head = nn.Linear(self.encoder_output_features, (self.n_players * self.n_roles))
        if self.predict in ['full']:
            # prediction of other players prediction of our own role
            self.role_backwards_prediction_head = nn.Linear(self.encoder_output_features, (self.n_players * self.n_roles))
        else:
            self.role_backwards_prediction_head = None

            # prediction of each observation
        c, h, w = self.obs_shape

        self.n_predictions = n_predictions

        if model.lower() in ["default", "fast"]:
            if model.lower() == "fast":
                print("warning: fast decoder is not implemented, using default.")
            decoder_fn = DefaultDecoder
        else:
            raise ValueError(f"Invalid model name {model}")

        self.observation_prediction_head = None
        self.observation_backwards_prediction_head = None
        self.action_prediction_head = None
        self.action_backwards_prediction_head = None

        if predict_observations:
            if self.predict in ['forward', 'full']:
                self.observation_prediction_head = decoder_fn(
                    (c, h * self.n_predictions, w),
                    self.encoder_output_features
                )
            if self.predict in ['full']:
                self.observation_backwards_prediction_head = decoder_fn(
                    (c, h * self.n_predictions, w),
                    self.encoder_output_features
                )

        if predict_actions:
            if self.predict in ['forward', 'full']:
                self.action_prediction_head = nn.Linear(self.encoder_output_features, (self.n_predictions * self.n_roles * self.n_actions))
            if self.predict in ['full']:
                self.action_backwards_prediction_head = nn.Linear(self.encoder_output_features, (self.n_predictions * self.n_roles * self.n_actions))

        self.set_device_and_dtype(self.device, self.dtype)

    def forward_sequence(self, obs, rnn_states, terminals=None):

        # check the input
        N, B, *obs_shape = obs.shape
        check_tensor("obs", obs, (N, B, *obs_shape), torch.uint8)
        check_tensor("rnn_states", rnn_states, (N, B, 2, self.memory_units), torch.float)
        if terminals is not None:
            check_tensor("terminals", terminals, (N, B), torch.int64)

        # upload to device
        obs = obs.to(self.device)
        rnn_states = rnn_states.to(self.device)
        if terminals is not None:
            terminals = terminals.to(self.device)

        result = {}

        encoder_output, new_rnn_states = self._forward_sequence(
            obs,
            rnn_states,
            terminals
        )

        # ------------------------------
        # role prediction
        # ------------------------------
        # these will come out as [N, B, n_players * n_roles] but we need [N, B, n_players, n_roles] for normalization
        unnormalized_role_predictions = self.role_prediction_head(encoder_output).reshape(
            [N, B, self.n_players, self.n_roles])
        role_prediction = torch.log_softmax(unnormalized_role_predictions, dim=-1)
        result['role_prediction'] = role_prediction

        if self.role_backwards_prediction_head is not None:
            unnormalized_role_backwards_predictions = self.role_backwards_prediction_head(encoder_output).reshape(
                [N, B, self.n_players, self.n_roles])
            role_backwards_prediction = torch.log_softmax(unnormalized_role_backwards_predictions, dim=-1)
            result['role_backwards_prediction'] = role_backwards_prediction

        # ------------------------------
        # obs forward prediction
        # ------------------------------

        def restack_observations(x: torch.Tensor):
            """
            Take input of (N*B, c, h*n_players, w) and return (N, B, n_players, c, h, w)
            :return:
            """
            c, h, w = self.obs_shape
            assert x.shape == (N*B, c, h*self.n_players, w)
            x = x.reshape(N, B, c, self.n_players * h, w)
            x = x.split(h, dim=3)
            x = torch.stack(x, dim=2)
            return x

        if self.observation_prediction_head is not None:
            # predictions will come out as (N*B, c, h*n_players, w)
            # but we need (N, B, n_players, c, h, w)
            obs_prediction = self.observation_prediction_head(encoder_output.reshape(N * B, self.encoder_output_features))
            result['obs_prediction'] = restack_observations(obs_prediction)

        # ------------------------------
        # obs backward prediction
        # ------------------------------

        if self.observation_backwards_prediction_head is not None:
            obs_pp = self.observation_backwards_prediction_head(encoder_output.reshape(N * B, self.encoder_output_features))
            result["obs_backwards_prediction"] = restack_observations(obs_pp)

        # ------------------------------
        # action forwards prediction
        # ------------------------------

        if self.action_prediction_head is not None:
            unnormalized_action_predictions = self.action_prediction_head(encoder_output.reshape(N * B, self.encoder_output_features))
            # result will be [N*B, n_predictions * n_roles * n_actions]
            unnormalized_action_predictions = unnormalized_action_predictions.reshape(N, B, self.n_predictions, self.n_roles, self.n_actions)
            action_predictions = torch.log_softmax(unnormalized_action_predictions, dim=-1)
            result["action_prediction"] = action_predictions

        if self.action_backwards_prediction_head is not None:
            unnormalized_action_backwards_predictions = self.action_backwards_prediction_head(encoder_output.reshape(N * B, self.encoder_output_features))
            # result will be [N*B, n_predictions * n_roles * n_actions]
            unnormalized_action_backwards_predictions = unnormalized_action_backwards_predictions.reshape(N, B, self.n_predictions, self.n_roles, self.n_actions)
            action_backwards_predictions = torch.log_softmax(unnormalized_action_backwards_predictions, dim=-1)
            result["action_backwards_prediction"] = action_backwards_predictions

        if self.nan_check:
            self._check_for_nans([(k,v) for k,v in result.items()] + [('rnn_states', new_rnn_states)])

        return result, new_rnn_states

class PolicyModel(BaseModel):

    """
    Multi-agent model for PPO Marl implementation

    This model provides the following outputs

    Feature Extractor -> LSTM -> Policy
                              -> Extrinsic Value (local estimation)
                              -> Intrinsic Value (local estimation)
                              -> Observation estimates for each player
                              -> Estimate of other players prediction of our own observation
    """

    def __init__(
            self,
            env: MultiAgentVecEnv,
            device="cpu",
            dtype=torch.float32,
            memory_units=128,
            out_features=128,
            model="default",
            data_parallel=False,
            roles=3,    # number of polcies / value_estimates to output
            lstm_mode="residual",
            nan_check=False,
            predict_roles=False,
    ):
        super().__init__(env, device, dtype, memory_units, out_features, model, data_parallel,
                         lstm_mode=lstm_mode, nan_check=nan_check)

        self.roles = roles

        # output heads
        self.policy_head = nn.Linear(self.encoder_output_features, self.roles * self.n_actions)
        torch.nn.init.xavier_uniform_(self.policy_head.weight, 0.01) # helps with exploration

        self.local_int_value_head = nn.Linear(self.encoder_output_features, roles)
        self.local_ext_value_head = nn.Linear(self.encoder_output_features, roles)
        # this is just here as an aux task. During voting it is important that the agent is aware of who is who
        if predict_roles:
            self.role_prediction_head = nn.Linear(self.encoder_output_features, (self.n_players * self.n_roles))
        else:
            self.role_prediction_head = None

        self.set_device_and_dtype(self.device, self.dtype)

    def forward_sequence(self, obs, rnn_states, roles=None, terminals=None):
        """
        Forwards sequence through model. If roles are provided returns the appropriate policy and value as
            log_policy: tensor [N, B, *policy_shape]
            ext_value: tensor [N, B]
            int_value: tensor [N, B]
        Regardless all role policy and values are returned as
            log_policies: tensor [N, B, R, *policy_shape]
            ext_values: tensor [N, B, R]
            int_values: tensor [N, B, R]

        Tensors will be uploaded to correct device

        :param obs:
        :param rnn_states:
        :param roles: Roles for each agent at each timestep (tensor) [N, B]
        :param terminals:
        :return: results dictionary, and updated rnn_states
        """

        # check the input
        (N, B, *obs_shape), device = obs.shape, obs.device
        check_tensor("obs", obs, (N, B, *obs_shape), torch.uint8)
        check_tensor("rnn_states", rnn_states, (N, B, 2, self.memory_units), torch.float32)
        if roles is not None:
            check_tensor("roles", roles, (N, B), torch.int64)
        if terminals is not None:
            check_tensor("terminals", terminals, (N, B), torch.int64)

        # upload to device
        obs = obs.to(self.device)
        rnn_states = rnn_states.to(self.device)
        if terminals is not None:
            terminals = terminals.to(self.device)
        if roles is not None:
            roles = roles.to(self.device)

        encoder_output, new_rnn_states = self._forward_sequence(obs, rnn_states, terminals)

        policy_outputs = self.policy_head(encoder_output).reshape(N, B, self.roles, self.n_actions)
        log_policy = torch.log_softmax(policy_outputs, dim=-1)
        ext_value = self.local_ext_value_head(encoder_output)
        int_value = self.local_int_value_head(encoder_output)

        result = {
            'role_log_policy': log_policy,
            'role_ext_value': ext_value,
            'role_int_value': int_value
        }

        # these will come out as [N, B, n_players * n_roles] but we need [N, B, n_players, n_roles] for normalization
        if self.role_prediction_head is not None:
            unnormalized_role_predictions = self.role_prediction_head(encoder_output).reshape(N, B, self.n_players,
                                                                                              self.n_roles)
            role_prediction = torch.log_softmax(unnormalized_role_predictions, dim=-1)
            result['policy_role_prediction'] = role_prediction

        def extract_roles(x, roles):
            """
            Input is N, B, R, *shape
            """
            N, B, R, *shape = x.shape
            assert roles.shape == (N, B)
            parts = []
            for n in range(N):
                parts.append(x[n:n+1, range(B), roles[n]])
            return torch.cat(parts, dim=0)

        if roles is not None:
            result['log_policy'] = extract_roles(log_policy, roles)
            result['ext_value'] = extract_roles(ext_value, roles)
            result['int_value'] = extract_roles(int_value, roles)

        if self.nan_check:
            self._check_for_nans([(k,v) for k,v in result.items()] + [('rnn_states', new_rnn_states)])

        return result, new_rnn_states

class SplitPolicyModel(nn.Module):
    """
    Separate model for each role.
    This is quite inefficient. A better solution would be to assign different models (or scripts) to the environment
    directly. That way each observation is processed once (not once per model as is the case here).
    """
    def __init__(
            self,
            env: MultiAgentVecEnv,
            device="cpu",
            dtype=torch.float32,
            memory_units=128,
            out_features=128,
            model="default",
            data_parallel=False,
            roles=3,    # number of policies / value_estimates to output
            lstm_mode="residual",
    ):

        super().__init__()
        self.n_roles = roles

        self.models = [
            PolicyModel(
                env=env,
                device=device,
                dtype=dtype,
                memory_units=memory_units,
                out_features=out_features,
                model=model,
                data_parallel=data_parallel,
                roles=roles,    # for compatability we have each model generate policies for all roles
                lstm_mode=lstm_mode,
                nan_check=False,
            ) for _ in range(self.n_roles)]

        # this allows Module to pick up on the models
        # it's a bit of a hack
        assert roles == 3
        self.model_0 = self.models[0]
        self.model_1 = self.models[1]
        self.model_2 = self.models[2]

        # setup variables so as to look like a normal PolicyModel
        self.input_dims = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.device = device
        self.dtype = dtype
        self.lstm_mode = lstm_mode
        self.n_agents = env.total_agents
        self.n_players = env.max_players
        self.obs_shape = env.observation_space.shape
        self.memory_units = memory_units
        self.encoder_output_features = self.models[0].encoder_output_features
        self.encoder_features = self.models[0].encoder_features
        self.encoder_type = self.models[0].encoder_type

    def forward_sequence(self, obs, rnn_states, roles=None, terminals=None):

        assert roles is not None, "split model requires roles to be specified at all times. "

        new_rnn_states = torch.zeros_like(rnn_states)

        result = {}


        for team_id, model in enumerate(self.models):
            # forward through model dedicated to this role
            # ideally we'd filter, but that is not possible as one segment may contain multiple roles (i.e
            # if it was reset half-way through
            # because of this we are duplicating the work 3 times
            results_part, rnn_states_part = model.forward_sequence(obs, rnn_states, roles, terminals)

            mask = (roles == team_id)

            for k, v in results_part.items():
                if k not in result: result[k] = torch.zeros_like(v)
                result[k][mask, ...] = v[mask, ...]

            rnn_mask = roles[-1] == team_id # use role of final transition
            new_rnn_states[rnn_mask] = rnn_states_part[rnn_mask]

        return result, new_rnn_states