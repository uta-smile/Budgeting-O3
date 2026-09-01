from __future__ import annotations

from unittest.mock import patch

import torch


def test_deterministic_pf_ode_consumes_explicit_latent_without_randomness() -> None:
    from boltz.model.modules import diffusionv2

    diffusion = object.__new__(diffusionv2.AtomDiffusion)
    torch.nn.Module.__init__(diffusion)
    # AtomDiffusion.device is derived from the score model's parameters.
    diffusion.score_model = torch.nn.Linear(1, 1)
    diffusion.gamma_min = 0.0
    diffusion.gamma_0 = 0.8
    diffusion.noise_scale = 1.0
    diffusion.step_scale = 1.0
    diffusion.num_sampling_steps = 3
    diffusion.training = False
    diffusion.step_scale_random = None
    diffusion.alignment_reverse_diff = False
    diffusion.sample_schedule = lambda steps: torch.tensor(
        [1.0, 0.5, 0.0], dtype=torch.float32
    )
    diffusion.preconditioned_network_forward = lambda coords, _time, network_condition_kwargs: coords * 0.5

    steering = {
        "fk_steering": False,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
    }
    atom_mask = torch.ones((1, 4), dtype=torch.bool)
    latent = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]
    )

    with patch.object(
        diffusionv2, "compute_random_augmentation", side_effect=AssertionError
    ), patch.object(torch, "randn", side_effect=AssertionError):
        result = diffusion.sample(
            atom_mask=atom_mask,
            num_sampling_steps=3,
            multiplicity=1,
            max_parallel_samples=1,
            steering_args=steering,
            initial_atom_coords=latent,
            deterministic=True,
        )

    output = result["sample_atom_coords"]
    assert output.shape == latent.shape
    assert torch.isfinite(output).all()
    assert not torch.equal(output, torch.zeros_like(output))


if __name__ == "__main__":
    test_deterministic_pf_ode_consumes_explicit_latent_without_randomness()
    print("custom sampler test passed")
