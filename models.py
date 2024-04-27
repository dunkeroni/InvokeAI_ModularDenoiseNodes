
from invokeai.invocation_api import (
    BaseInvocationOutput,
    BaseInvocation,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)
from pydantic import BaseModel, Field

class ModuleData(BaseModel): #use child classes for different module types
    name: str = Field(description="Name of the module")
    module_type: str = Field(description="Type of the module")
    module: str = Field(description="Name of the module function")
    module_kwargs: dict | None = Field(description="Keyword arguments to pass to the module function")

class NP_ModuleData(ModuleData):
    module_type: str = Field(default="noise_pred", description="Type of the module")

class PreG_ModuleData(ModuleData):
    module_type: str = Field(default="pre_noise_guidance", description="Type of the module")

class PoG_ModuleData(ModuleData):
    module_type: str =  Field(default="post_noise_guidance", description="Type of the module")

@invocation_output("module_data_output")
class ModuleDataOutput(BaseInvocationOutput):
    module_data_output: ModuleData | None = OutputField(
        title="Module Data Output",
        description="Information for calling the module in denoise latents step()"
    )
@invocation_output("np_module_data_output")
class NP_ModuleDataOutput(ModuleDataOutput):
    module_data_output: ModuleData | None = OutputField(
        title="[NP] Module Data",
        description="Noise Prediction module data"
    )
@invocation_output("preg_module_data_output")
class PreG_ModuleDataOutput(ModuleDataOutput):
    module_data_output: ModuleData | None = OutputField(
        title="[PreG] Module Data",
        description="PRE-Noise Guidance module data"
    )
@invocation_output("pog_module_data_output")
class PoG_ModuleDataOutput(ModuleDataOutput):
    module_data_output: ModuleData | None = OutputField(
        title="[PoG] Module Data",
        description="POST-Noise Guidance module data"
    )

@invocation_output("module_data_collection_output")
class ModuleDataCollectionOutput(BaseInvocationOutput):
    module_data_output: list[ModuleData] | None = OutputField(
        title="Collected Module Data",
        description="Information for calling the modules in denoise latents",
        default=None
    )

@invocation("module_collection",
            title="Module Collection",
            tags=["modular", "collection"],
            category="modular",
            version="1.0.0"
            )
class ModuleCollectionInvocation(BaseInvocation):
    """Collects multiple module types together"""
    pre_noise_module: ModuleData = InputField(
        default=None,
        description="Information for calling the module in denoise latents step()",
        title="[PreG] Module Data",
        input=Input.Connection,
    )
    noise_pred_module: ModuleData = InputField(
        default=None,
        description="Information for calling the module in denoise latents step()",
        title="[NP] Module Data",
        input=Input.Connection,
    )
    pog_noise_module: ModuleData = InputField(
        default=None,
        description="Information for calling the module in denoise latents step()",
        title="[PoG] Module Data",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> ModuleDataCollectionOutput:
        #make sure list is always list[ModuleData] and does not included Nones
        modules_list = []
        if self.pre_noise_module is not None:
            modules_list.append(self.pre_noise_module)
        if self.noise_pred_module is not None:
            modules_list.append(self.noise_pred_module)
        if self.pog_noise_module is not None:
            modules_list.append(self.pre_noise_module)
        
        if modules_list == []:
            return ModuleDataCollectionOutput(
                module_data_output=None
            )
        else:
            return  ModuleDataCollectionOutput(
                module_data_output=modules_list
            )