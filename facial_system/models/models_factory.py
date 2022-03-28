import sys
sys.path.append('./')
import models as model_md
import logging
import os
# from demo_image import load_model_classify


class CelebIndexModels:
    def __init__(self, models_cfg, device, mode='deploy'):
        self.logger = logging.getLogger(os.environ['LOGGER_ID'])
        w_dir = os.environ['WEIGHTS_PATH']
        
        # detection model
        detection_cfg = models_cfg['detection']
        if mode != 'deploy':
            detection_cfg['args']['backbone_path'] = os.path.join(w_dir, 
                                                        detection_cfg['args']['backbone_path'])
            detection_cfg['args']['checkpoint_path'] = os.path.join(w_dir, 
                                                        detection_cfg['args']['checkpoint_path'])

        self.detection_md = getattr(model_md, detection_cfg['name'])(**detection_cfg['args'])
        self.detection_md.load_checkpoint(device)
        self.logger.info('Loading detection model {} done ...'.format(detection_cfg['name']))

        # embedding model
        embedding_cfg = models_cfg['encoder']
        if mode != 'deploy':
            embedding_cfg['args']['checkpoint_path'] = os.path.join(w_dir, 
                                                        embedding_cfg['args']['checkpoint_path'])
        
        self.emb_model = getattr(model_md, embedding_cfg['name'])(**embedding_cfg['args'])
        self.emb_model.load_checkpoint(device)
        self.logger.info('Loading embedding model {} done ...'.format(embedding_cfg['name']))

        # emotion model
        emotion_cfg = models_cfg['emotion']
        if mode != 'deploy':
            emotion_cfg['args']['checkpoint_path'] = os.path.join(w_dir, 
                                                        emotion_cfg['args']['checkpoint_path'])
        
        self.emt_model = getattr(model_md, emotion_cfg['name'])(**emotion_cfg['args'])
        self.emt_model.load_checkpoint(device)
        self.logger.info('Loading emotion model {} is done ...'.format(emotion_cfg['name']))

        # mlps models
        mlp_cfg = models_cfg['mlp']
        mlp_args = mlp_cfg['args']
        self.classify_models = []
        for idx, path in enumerate(mlp_cfg['paths']):
            if mode != 'deploy':
                path = os.path.join(w_dir, path)
            classify_model = model_md.MLPModel(**mlp_args, checkpoint_path=path)
            classify_model.load_checkpoint(device)
            self.classify_models.append(classify_model)
        self.logger.info('Loading mlp models done ...')
    
    def get_models(self):
        return {
            'face_dt_model': self.detection_md,
            'emb_model': self.emb_model,
            'mlp_models': self.classify_models,
            'emt_model': self.emt_model
        }