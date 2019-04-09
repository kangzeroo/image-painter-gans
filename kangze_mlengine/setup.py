from setuptools import find_packages, setup

REQUIRED_PACKAGES = ['tensorflow-gpu==2.0.0a0',
                     'CUDA==10.0',
                     'Keras==2.2.4',
                     'h5py==2.9.0',
                     'numpy==1.16.2',
                     'Pillow==2.2.1',
                     'opencv-python==4.0.0.21',
                     'google==2.0.1',
                     'google-cloud-storage==1.14.0',
                     'google-cloud==0.34.0',
                     'google-api-python-client==1.7.8']
# REQUIRED_PACKAGES = ['tensorflow-gpu==2.0.0a0',
#                      'Keras==2.2.4',
#                      'h5py',
#                      'numpy',
#                      'Pillow',
#                      'opencv-python',
#                      'google',
#                      'google-cloud-storage',
#                      'google-cloud',
#                      'google-api-python-client']

setup(
    name='messy_room_ai',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)
