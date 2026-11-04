import segmentation_models_pytorch as smp


def build_unet(encoder="resnet34", pretrained=True):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=2,      # VV, VH
        classes=1,          # one logit per pixel: water vs not
    )