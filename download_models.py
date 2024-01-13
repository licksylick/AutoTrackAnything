from mega import Mega
import zipfile
import os

mega = Mega()
m = mega.login()
m.download_url('https://mega.nz/file/hTwWhDwY#BzBZ8xnsHZjzhm0XjpVE50YNFlj3_G7rb5Bd_1gVknc')
with zipfile.ZipFile('AutoTrackAnything_models.zip', 'r') as zip_ref:
    zip_ref.extractall()
os.remove('AutoTrackAnything_models.zip')

print('Success! Models have been downloaded to the "saves" folder')
