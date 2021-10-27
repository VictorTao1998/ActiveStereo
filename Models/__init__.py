from .ActiveStereoNet import ActiveStereoNet

def get_model(config):

    max_disp = config.ARGS.MAX_DISP
    scale_factor = 8
    img_shape = [config.ARGS.CROP_WIDTH, config.ARGS.CROP_HEIGHT]
    model = ActiveStereoNet(max_disp, scale_factor, img_shape)

    return model