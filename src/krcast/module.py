
import tensorflow.Module as Module
import logging


class MyModule(Module):

    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, #context=tf.Module.context.gpu(), 
                 work_load_list=None,
                 fixed_param_names=None, state_names=None, name=None):
        self._name = name
        super(MyModule, self).__init__(symbol=symbol,
                                       data_names=data_names,
                                       label_names=label_names,
                                       logger=logger,
                                       ##context=context,
                                       work_load_list=work_load_list,
                                       fixed_param_names=fixed_param_names,
                                       state_names=state_names)
        self._tmp_grads = None

