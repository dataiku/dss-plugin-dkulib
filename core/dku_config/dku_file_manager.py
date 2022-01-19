from dataiku.customrecipe import get_input_names_for_role, get_output_names_for_role
import dataiku
from .dku_config import DkuConfig


class DkuFileManager(DkuConfig):
    """
    Use this class to create an object that contains the different input and output datasets/folders of a custom recipe
    
    Usage example:

    .. code-block:: python

        file_manager = DkuFileManager()
        # add a required and an optional input dataset
        file_manager.add_input_dataset("input_dataset")
        file_manager.add_input_dataset("optional_input_dataset", required=False)
        # add a required output dataset and an optional output folder
        file_manager.add_output_dataset("output_dataset")
        file_manager.add_output_folder("optional_output_folder", required=False)
    """    
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

    def add_file(self, side, type_, role, **kwargs):
        file = DkuFileManager._retrieve_file_from_dss(side, type_, role)
        self.add_param(name=role, value=file, **kwargs)

    def add_input_folder(self, role, required=True):
        self.add_file("input", "folder", role, required=required)

    def add_output_folder(self, role, required=True):
        self.add_file("output", "folder", role, required=required)

    def add_input_dataset(self, role, required=True):
        self.add_file("input", "dataset", role, required=required)

    def add_output_dataset(self, role, required=True):
        self.add_file("output", "dataset", role, required=required)

    @staticmethod
    def _retrieve_file_from_dss(side, type_, role):
        dku_func = get_input_names_for_role if side == "input" else get_output_names_for_role
        dku_type = dataiku.Folder if type_ == "folder" else dataiku.Dataset
        roles = dku_func(role)
        return dku_type(roles[0]) if roles else None

    @staticmethod
    def write_to_folder(folder, file_path, content):
        with folder.get_writer(file_path) as w:
            w.write(content.encode())
