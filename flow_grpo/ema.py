import mindspore as ms
import mindspore.nn as nn


class EMAModuleWrapper(nn.Cell):

    def __init__(
        self,
        parameters: ms.ParameterTuple,
        decay: float = 0.9999,
        update_step_interval: int = 1,
    ):
        super().__init__()
        self.ema_parameters = parameters.clone("ema")
        self.temp_stored_parameters = None
        self.decay = decay
        self.update_step_interval = update_step_interval

    def get_current_decay(self, optimization_step: int) -> float:
        return min((1 + optimization_step) / (10 + optimization_step),
                   self.decay)

    def construct(self, parameters: ms.ParameterTuple,
                  optimization_step: int) -> None:
        one_minus_decay = 1 - self.get_current_decay(optimization_step)

        if (optimization_step + 1) % self.update_step_interval == 0:
            for ema_parameter, parameter in zip(self.ema_parameters,
                                                parameters,
                                                strict=True):
                if parameter.requires_grad:
                    ema_parameter.add_(one_minus_decay *
                                       (parameter - ema_parameter))

    def copy_ema_to(self,
                    parameters: ms.ParameterTuple,
                    store_temp: bool = True) -> None:
        """
        Copy the EMA parameters to the provided parameters.
        """
        if store_temp:
            self.temp_stored_parameters = parameters.clone("tmp")

        parameters = list(parameters)
        for ema_parameter, parameter in zip(self.ema_parameters,
                                            parameters,
                                            strict=True):
            parameter.copy_(ema_parameter)

    def copy_temp_to(self, parameters: ms.ParameterTuple) -> None:
        """
        Copy the temporarily stored parameters back to the original parameters.
        This is used to restore the original parameters after an EMA update.
        """
        assert self.temp_stored_parameters is not None
        for temp_parameter, parameter in zip(self.temp_stored_parameters,
                                             parameters,
                                             strict=True):
            parameter.copy_(temp_parameter)

        self.temp_stored_parameters = None
