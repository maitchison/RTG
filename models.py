import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel as DP
import math

import utils

from marl_env import MultiAgentVecEnv

class BaseEncoder(nn.Module):

    def __init__(self, input_dims, out_features):
        """
        :param input_dims: intended dims of input, excluding batch size (height, width, channels)
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

        self.final_dims = (64, self.input_dims[0]//2//2//2, self.input_dims[1]//2//2//2)

        self.conv1 = nn.Conv2d(self.input_dims[2], 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(utils.prod(self.final_dims), self.out_features)

        print(f" -created default encoder, final dims {self.final_dims}")

    def forward(self, x):
        """
        input float32 tensor of dims [b, h, w, c]
        return output tensor of dims [b, d], where d is the number of units in final layer.
        """
        assert type(x) == torch.Tensor, f"Input must be torch tensor not {type(x)}"
        assert x.shape[1:] == self.input_dims, f"Input dims {x.shape[1:]} must match {self.input_dims}"
        assert x.dtype == torch.float32, f"Datatype should be torch.float32 not {x.dtype}"

        b = x.shape[0]

        # put in BCHW format for pytorch.
        x = x.transpose(2, 3)
        x = x.transpose(1, 2)

        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        assert x.shape[1:] == self.final_dims, f"Expected final shape to be {self.final_dims} but found {x.shape[1:]}"
        x = x.reshape((b, -1))
        x = torch.relu(self.fc(x))

        return x

class LargeEncoder(BaseEncoder):
    """ Encodes from observation to hidden representation. """

    def __init__(self, input_dims, out_features=128):
        """
        :param input_dims: intended dims of input, excluding batch size (height, width, channels)
        """
        super().__init__(input_dims, out_features)

        self.final_dims = (128, self.input_dims[0]//2//2, self.input_dims[1]//2//2)

        self.conv1 = nn.Conv2d(self.input_dims[2], 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(utils.prod(self.final_dims), self.out_features)

        print(f" -created large encoder, final dims {self.final_dims}")

    def forward(self, x):
        """
        input float32 tensor of dims [b, h, w, c]
        return output tensor of dims [b, d], where d is the number of units in final layer.
        """
        assert type(x) == torch.Tensor, f"Input must be torch tensor not {type(x)}"
        assert x.shape[1:] == self.input_dims, f"Input dims {x.shape[1:]} must match {self.input_dims}"
        assert x.dtype == torch.float32, f"Datatype should be torch.float32 not {x.dtype}"

        b = x.shape[0]

        # put in BCHW format for pytorch.
        x = x.transpose(2, 3)
        x = x.transpose(1, 2)

        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        assert x.shape[1:] == self.final_dims, f"Expected final shape to be {self.final_dims} but found {x.shape[1:]}"
        x = x.reshape((b, -1))
        x = torch.relu(self.fc(x))

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

        self.final_dims = (64, self.input_dims[0]//6, self.input_dims[1]//6)

        # based on nature

        self.conv1 = nn.Conv2d(self.input_dims[2], 32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(utils.prod(self.final_dims), self.out_features)

        print(f" -created fast encoder, final dims {self.final_dims}")

    def forward(self, x):
        """
        input float32 tensor of dims [b, h, w, c]
        return output tensor of dims [b, d], where d is the number of units in final layer.
        """

        assert x.shape[1:] == self.input_dims, f"Input dims {x.shape[1:]} must match {self.input_dims}"
        assert x.dtype == torch.float32, f"Datatype should be torch.float32 not {x.dtype}"

        b = x.shape[0]

        # put in BCHW format for pytorch.
        x = x.transpose(2, 3)
        x = x.transpose(1, 2)

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
        x = torch.sigmoid(self.deconv3(x))

        # the decoder doubles the size each time, edges tend not to predict well anyway due to padding
        # so we calculate the required dims (rx,ry) and the excess (ex,ry), then take a center crop
        rx, ry = self.output_dims[-1], self.output_dims[-2]
        ex, ey = x.shape[-1] - rx, x.shape[-2] - ry
        x = x[:, :, ey//2:ey//2+ry, ex//2:ex//2+rx]

        return x


class LargeDecoder(BaseDecoder):
    """ Decodes from hidden representation to observation. """

    def __init__(self, output_dims, hidden_features=128):
        """
        :param output_dims: (c,h,w)
        :param hidden_features:
        """

        initial_dims = (128, math.ceil((output_dims[1] + 2) / 4), math.ceil((output_dims[2] + 2) / 4))
        super().__init__(output_dims, initial_dims, hidden_features)

        self.fc = nn.Linear(self.hidden_features, utils.prod(self.initial_dims))
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, padding=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2)
        self.deconv4 = nn.ConvTranspose2d(32, self.output_dims[0], kernel_size=4, stride=2, padding=1)

        print(f" -created large decoder, initial dims {self.initial_dims}")

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
        x = torch.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

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
            residual_connection=True
    ):
        assert env.observation_space.dtype == np.uint8, "Observation space should be 8bit integer"

        if residual_connection:
            assert memory_units == out_features

        super().__init__()

        self.input_dims = env.observation_space.shape
        self.actions = env.action_space.n
        self.device = device
        self.dtype = dtype
        self.residual_connection = residual_connection

        if model.lower() == "default":
            self.encoder = DefaultEncoder(env.observation_space.shape, out_features=out_features)
        elif model.lower() == "fast":
            self.encoder = FastEncoder(env.observation_space.shape, out_features=out_features)
        elif model.lower() == "large":
            self.encoder = LargeEncoder(env.observation_space.shape, out_features=out_features)
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

        self._encoder = self.encoder # the encoder before and data_parallel was applied
        if data_parallel:
            # enable multi gpu :)
            print(f" -enabling {utils.Color.OKGREEN}Multi-GPU{utils.Color.ENDC} support")
            self.encoder = DP(self.encoder)

    def forward_sequence(self, obs, rnn_states, terminals=None):
        raise NotImplemented()

    def _forward_sequence(self, obs, rnn_states, terminals=None):
        """
        Forward a sequence of observations through model, returns dictionary of outputs.
        :param obs: input observations, tensor of dims [N, B, observation_shape], which should be channels last. (BHWC)
        :param rnn_states: float32 tensor of dims [B, 2, memory_dims] containing the initial rnn h,c states
        :param terminals: (optional) tensor of dims [N, B] indicating timesteps that are terminal
        :return: lstm outputs

        """

        N, B, *obs_shape = obs.shape

        # merge first two dims into a batch, run it through encoder, then reshape it back into the correct form.
        assert tuple(obs_shape) == self.obs_shape
        assert tuple(rnn_states.shape) == (B, 2, self.memory_units), f"Expected {(B, 2, self.memory_units)} found {rnn_states.shape}."
        if terminals is not None:
            assert tuple(terminals.shape) == (N, B)

        if terminals is not None and terminals.sum() == 0:
            # just ignore terminals there aren't any (faster)
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
        new_rnn_states[:, 0, :] = h
        new_rnn_states[:, 1, :] = c

        if self.residual_connection:
            return lstm_output + encoding
        else:
            return lstm_output

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


class DeceptionModel(BaseModel):

    def __init__(
            self,
            env: MultiAgentVecEnv,
            device="cpu",
            dtype=torch.float32,
            memory_units=512,
            out_features=512,
            model="default",
            data_parallel=False,
            residual_connection=True
    ):
        super().__init__(env, device, dtype, memory_units, out_features, model, data_parallel, residual_connection)

        # prediction of each role, in public_id order
        self.role_prediction_head = nn.Linear(self.memory_units, (self.n_players * self.n_roles))
        # prediction of each observation in public_id order
        h, w, c = self.obs_shape

        if model.lower() in ["default", "fast"]:
            if model.lower() == "fast":
                print("warning: fast decoder is not implemented, using default.")
            self.observation_prediction_head = DefaultDecoder(
                (c, h * self.n_players, w),
                self.memory_units
            )
        elif model.lower() == "large":
            self.observation_prediction_head = LargeDecoder(
                (c, h * self.n_players, w),
                self.memory_units
            )
        else:
            raise ValueError(f"Invalid model name {model}")

        if data_parallel:
            self.observation_prediction_head = DP(self.observation_prediction_head)

        self.set_device_and_dtype(self.device, self.dtype)

    def forward_sequence(self, obs, rnn_states, terminals=None):

        N, B, *obs_shape = obs.shape
        encoder_output, new_rnn_states, encodings = self._forward_sequence(
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

        # ------------------------------
        # obs prediction
        # ------------------------------

        obs_prediction = self.observation_prediction_head(encoder_output.reshape(N * B, self.memory_units))

        # predictions will come out as (N*B, c, h*n_players, w)
        # but we need (N, B, n_players, h, w, c)
        h, w, c = self.obs_shape
        obs_prediction = obs_prediction.reshape(N, B, c, self.n_players * h, w)
        obs_prediction = obs_prediction.split(h, dim=3)
        obs_prediction = torch.stack(obs_prediction, dim=2)
        obs_prediction = obs_prediction.transpose(-3, -1)
        obs_prediction = obs_prediction.transpose(-3, -2)

        result = {}
        result['role_prediction'] = role_prediction
        result['obs_prediction'] = obs_prediction
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
            residual_connection=True
    ):
        super().__init__(env, device, dtype, memory_units, out_features, model, data_parallel, residual_connection)

        # output heads
        self.policy_head = nn.Linear(self.memory_units, env.action_space.n)
        torch.nn.init.xavier_uniform_(self.policy_head.weight, 0.01) # helps with exploration

        self.local_int_value_head = nn.Linear(self.memory_units, 1)
        self.local_ext_value_head = nn.Linear(self.memory_units, 1)

        self.set_device_and_dtype(self.device, self.dtype)

    def forward_sequence(self, obs, rnn_states, terminals=None):

        N, B, *obs_shape = obs.shape
        lstm_output, new_rnn_states = self._forward_sequence(obs, rnn_states, terminals)

        log_policy = torch.log_softmax(self.policy_head(lstm_output), dim=-1)
        ext_value = self.local_ext_value_head(lstm_output).squeeze(dim=-1)
        int_value = self.local_int_value_head(lstm_output).squeeze(dim=-1)

        result = {
            'log_policy': log_policy,
            'ext_value': ext_value,
            'int_value': int_value
        }

        return result, new_rnn_states