import parameterized_transforms.core as ptc


class ParameterFreeTransform(ptc.AtomicTransform):
    def __init__(
        self,
        tx_mode: ptc.TRANSFORM_MODE_TYPE = ptc.TransformMode.CASCADE,
    ) -> None:
        super().__init__(tx_mode=tx_mode)
        self.param_count = self.set_param_count()

    def set_param_count(self) -> int:
        return 0

    def get_raw_params(self, img: ptc.IMAGE_TYPE) -> ptc.PARAM_TYPE:
        return ()

    def post_process_params(
        self, img: ptc.IMAGE_TYPE, params: ptc.PARAM_TYPE
    ) -> ptc.PARAM_TYPE:
        return params

    def extract_params(
        self, params: ptc.PARAM_TYPE
    ) -> tuple[ptc.PARAM_TYPE, ptc.PARAM_TYPE]:
        return (), params

    def pre_process_params(
        self, img: ptc.IMAGE_TYPE, params: ptc.PARAM_TYPE
    ) -> ptc.PARAM_TYPE:
        return params

    def get_default_params(
        self, img: ptc.IMAGE_TYPE, processed: bool = True
    ) -> ptc.PARAM_TYPE:
        raw_id_params = ()
        return (
            self.post_process_params(img=img, params=raw_id_params)
            if processed
            else raw_id_params
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"param_count={self.param_count}, "
            f"tx_mode={self.tx_mode}"
            f")"
        )


def make_parameter_free(cls):
    class Wrapper(ParameterFreeTransform):
        def __init__(
            self,
            *cls_args,
            tx_mode: ptc.TRANSFORM_MODE_TYPE = ptc.TransformMode.CASCADE,
            **cls_kwargs,
        ):
            super().__init__(tx_mode=tx_mode)
            self.transform = cls(*cls_args, **cls_kwargs)

        def apply_transform(self, img, params, **kwargs):
            return self.transform(img)

    return Wrapper
