from src.models.components.partmae_v6 import PARTMaskedAutoEncoderViT


def test_freeze_zero_weight_losses():
    model = PARTMaskedAutoEncoderViT(
        lambda_pose=0.0,
        lambda_pmatch=0.0,
        lambda_pcr=0.0,
        lambda_cinv=0.0,
        lambda_ccr=0.0,
        lambda_cosa=0.0,
    )

    assert all(not p.requires_grad for p in model._pose_loss.parameters())
    assert all(not p.requires_grad for p in model._pmatch_loss.parameters())
    assert all(not p.requires_grad for p in model._pcr_loss.parameters())
    assert all(not p.requires_grad for p in model._cinv_loss.parameters())
    assert all(not p.requires_grad for p in model._ccr_loss.parameters())
    assert all(not p.requires_grad for p in model._cosa_loss.parameters())

    assert not any(p.requires_grad for p in model.get_decoder_params())


def test_param_group_helpers():
    model = PARTMaskedAutoEncoderViT()

    enc_params = list(model.get_encoder_params())
    dec_params = list(model.get_decoder_params())

    assert enc_params and dec_params
    assert not set(enc_params).intersection(dec_params)
