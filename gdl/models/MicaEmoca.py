from .DECA import * 
from .mica.mica import *


class MicaEmocaModule(LightningModule):

    def __init__(self, model_params, learning_params, inout_params, stage_name = ""):
        super().__init__()

        #TODO: MICA uses FLAME  
        # 1) This is redundant - get rid of it 
        # 2) Make sure it's the same FLAME as EMOCA

    def _instantiate_deca(self, model_params):
        """
        Instantiate the DECA network.
        """
        # which type of DECA network is used
        if 'deca_class' not in model_params.keys() or model_params.deca_class is None:
            print(f"Deca class is not specified. Defaulting to {str(DECA.__class__.__name__)}")
            # vanilla DECA by default (not EMOCA)
            deca_class = DECA
        else:
            # other type of DECA-inspired networks possible (such as ExpDECA, which is what EMOCA)
            deca_class = class_from_str(model_params.deca_class, sys.modules[__name__])

        # instantiate the network
        self.deca = deca_class(config=model_params)


class EMICA(ExpDECA): 

    def __init__(self, config):
        super().__init__(config)
        self.E_mica = Mica()

    

    