from platformdirs import user_data_dir,user_cache_dir
from dataclasses import dataclass
import os
import pooch
import zipfile

WEIGHTS_BASE_URL='https://github.com/YaoYinYing/ThermoMPNN-D/releases/download/v1.0.0/'


@dataclass(frozen=True)
class ModelFetchSetting:
    name: str
    version: str
    url: str
    md5sum: str

    @property
    def basename(self):
        return os.path.basename(self.url).rstrip('.zip')
    
    @property
    def weight_path(self):
        return os.path.join(user_data_dir(self.name,version=self.version, ensure_exists=True), self.basename)
    
    @property
    def ready(self):
        return os.path.exists(self.weight_path) and os.listdir(self.weight_path)
    
    def setup(self):
        if self.ready:
            print(f'Already downloaded {self.basename} to {self.weight_path}')
            return self.weight_path
        
        print(f'Downloading {self.basename}...')
        downloaded=pooch.retrieve(self.url,known_hash=f'md5:{self.md5sum}', path=user_cache_dir(f'downloading_{self.name}_weights', ensure_exists=True), progressbar=True)
        dist_dir=os.path.dirname(self.weight_path)
        expanded_dirs = os.listdir(dist_dir)
        if not expanded_dirs:
            
            print(f'Extracting {downloaded} to {dist_dir}')

            with zipfile.ZipFile(downloaded, mode="r") as z:
                z.extractall(path=dist_dir)

        extracted_files = os.listdir(dist_dir)
        print(f'Extracted {extracted_files}')
        return self.weight_path


thermompnn_weigths=ModelFetchSetting(
    name='ThermoMPNN',
    version='double',
    url=WEIGHTS_BASE_URL+'model_weights.zip',
    md5sum='a295c4a43d197724c626d5dd38fccb2c'
)
vanilla_weigths=ModelFetchSetting(
    name='ProteinMPNN',
    version='vanilla',
    url=WEIGHTS_BASE_URL+'vanilla_model_weights.zip',
    md5sum='2647210cf948468c5a6b27f993a58744'
)

def ensure_weights():
    for m in (thermompnn_weigths, vanilla_weigths):
        m.setup()


