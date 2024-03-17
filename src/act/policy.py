import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from detr.models.backbone import FilMedBackbone


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")
        multi_gpu = args_override["multi_gpu"]
        self.num_queries = (
            self.model.module.num_queries if multi_gpu else self.model.num_queries
        )

    def __call__(self, qpos, image, actions=None, is_pad=None, command_embedding=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos,
                image,
                env_state,
                actions,
                is_pad,
                command_embedding=command_embedding,
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(
                qpos, image, env_state, command_embedding=command_embedding
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


# Not used in this paper. But leave Diffusion Policy integration here in case anyone is interested.
# Requires installing https://github.com/ARISE-Initiative/robomimic/tree/r2d2 in src (such that src/robomimic is a valid path)
# Remember to install extra dependencies: $ pip install -e src/robomimic
class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
        from robomimic.algo.diffusion_policy import (
            replace_bn_with_gn,
            ConditionalUnet1D,
        )
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        super().__init__()

        self.camera_names = args_override["camera_names"]
        self.observation_horizon = args_override["observation_horizon"]  # TODO
        self.action_horizon = args_override["action_horizon"]  # apply chunk size
        self.prediction_horizon = args_override["prediction_horizon"]  # chunk size
        self.num_inference_timesteps = args_override["num_inference_timesteps"]
        self.ema_power = args_override["ema_power"]
        self.lr = args_override["lr"]
        self.weight_decay = 0
        backbone_name = args_override["backbone"]
        use_language = "film" in backbone_name

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override["action_dim"]
        self.obs_dim = (
            self.feature_dimension * len(self.camera_names) + 14
        )  # camera features and proprio
        if use_language:
            self.obs_dim += self.feature_dimension
            self.lang_embed_proj = nn.Linear(
                768, self.feature_dimension
            )  # TODO: might change
            in_shape = [1536, 10, 10]
        else:
            in_shape = [512, 15, 20]

        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            if use_language:
                print("Using FiLMed backbone.")
                backbone = FilMedBackbone(backbone_name)
            else:
                print("Using ResNet18Conv backbone.")
                backbone = ResNet18Conv(
                    **{
                        "input_channel": 3,
                        "pretrained": False,
                        "input_coord_conv": False,
                    }
                )
            backbones.append(backbone)
            pools.append(
                SpatialSoftmax(
                    **{
                        "input_shape": in_shape,
                        "num_kp": self.num_kp,
                        "temperature": 1.0,
                        "learnable_temperature": False,
                        "noise_std": 0.0,
                    }
                )
            )
            linears.append(
                torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension)
            )
        self.backbones = nn.ModuleList(backbones)
        self.pools = nn.ModuleList(pools)
        self.linears = nn.ModuleList(linears)

        self.backbones = replace_bn_with_gn(self.backbones)  # TODO

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim * self.observation_horizon,
        )

        nets = nn.ModuleDict(
            {
                "policy": nn.ModuleDict(
                    {
                        "backbones": self.backbones,
                        "pools": self.pools,
                        "linears": self.linears,
                        "noise_pred_net": self.noise_pred_net,
                    }
                )
            }
        )

        if use_language:
            nets["policy"]["lang_embed_proj"] = self.lang_embed_proj

        nets = nets.float()
        if args_override["multi_gpu"] and not args_override["is_eval"]:
            assert torch.cuda.device_count() > 1
            print(f"Using {torch.cuda.device_count()} GPUs")
            nets = torch.nn.DataParallel(nets)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nets.to(device)
        self.nets = nets

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        # count parameters
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("number of trainable parameters: %.2fM" % (n_parameters / 1e6,))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None, command_embedding=None):
        B = qpos.shape[0]
        if actions is not None:  # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets["policy"]["backbones"][cam_id](
                    cam_image, command_embedding
                )
                pool_features = nets["policy"]["pools"][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets["policy"]["linears"][cam_id](pool_features)
                all_features.append(out_features)

            if command_embedding is None:
                obs_cond = torch.cat(all_features + [qpos], dim=1)
            else:
                command_embedding_proj = nets["policy"]["lang_embed_proj"](
                    command_embedding
                )
                obs_cond = torch.cat(
                    all_features + [qpos] + [command_embedding_proj], dim=1
                )

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=obs_cond.device,
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # predict the noise residual
            noise_pred = nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond
            )

            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction="none")
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict["l2_loss"] = loss
            loss_dict["loss"] = loss
            return loss_dict
        else:  # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim

            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets["policy"]["backbones"][cam_id](
                    cam_image, command_embedding
                )
                pool_features = nets["policy"]["pools"][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets["policy"]["linears"][cam_id](pool_features)
                all_features.append(out_features)

            if command_embedding is None:
                obs_cond = torch.cat(all_features + [qpos], dim=1)
            else:
                command_embedding_proj = nets["policy"]["lang_embed_proj"](
                    command_embedding
                )
                obs_cond = torch.cat(
                    all_features + [qpos] + [command_embedding_proj], dim=1
                )

            # initialize action from Guassian noise
            noisy_action = torch.randn((B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets["policy"]["noise_pred_net"](
                    sample=naction, timestep=k, global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
        }

    def deserialize(self, model_dict):
        if "nets" in model_dict:
            status = self.nets.load_state_dict(model_dict["nets"])
        else:  # backward compatibility
            nets_dict = {}
            for k, v in model_dict.items():
                if k.startswith("nets."):
                    nets_dict[k[5:]] = v
            status = self.nets.load_state_dict(nets_dict)
        print("Loaded diffusion model")
        return status


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
