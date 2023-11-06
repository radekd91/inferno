from omegaconf import open_dict


def sequence_decoder_from_cfg(cfg):
    decoder_cfg = cfg.model.sequence_decoder
    if decoder_cfg is None or decoder_cfg.type in ["none", "no"]:
        return None
    elif decoder_cfg.type == "LinearDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import LinearDecoder
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = LinearDecoder(decoder_cfg)
    # elif decoder_cfg.type == "LinearAutoRegDecoder":
    #     from inferno.models.talkinghead.FaceFormerDecoder import LinearAutoRegDecoder
    #     with open_dict(decoder_cfg):
    #         decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
    #         decoder_cfg.predict_exp = cfg.model.output.predict_expcode
    #         decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
    #     decoder = LinearAutoRegDecoder(decoder_cfg)
    elif decoder_cfg.type == "BertDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import BertDecoder 
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = BertDecoder(decoder_cfg)

    elif decoder_cfg.type == "MLPDecoder":
        from inferno.models.talkinghead.FaceFormerDecoder import MLPDecoder 
        with open_dict(decoder_cfg):
            decoder_cfg.num_training_subjects = len(cfg.data.train_subjects)
            decoder_cfg.predict_exp = cfg.model.output.predict_expcode
            decoder_cfg.predict_jaw = cfg.model.output.predict_jawpose
        decoder = MLPDecoder(decoder_cfg)
    else: 
        raise ValueError(f"Unknown sequence decoder model type '{decoder_cfg.type}'")

    return decoder