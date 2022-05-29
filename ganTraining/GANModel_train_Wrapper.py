import matplotlib.pyplot as plt
import spectral.io.envi as envi
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config=tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction=0.50
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
set_session(tf.Session(config=config))

'Import class with Deep Models'
from GANModel_1d import GANModel_1d
from GANModel_Train import GANModel_Train


hdrName='/Volume1/data/CRISM/yuki/sabcondv4/jezero/HRL000040FF/HRL000040FF_07_IF183L_TRR3_atcr_sabcondv4_1_Lib11123_1_4_5_l1_gadmm_a_v2_ca_ice_b200_nr.hdr'
hdr = envi.read_envi_header(hdrName)



if 'wavelength' in hdr:
    try:
        wavelength = [float(b) for b in hdr['wavelength']]
    except:
        pass
wavelength = wavelength[4:244]
del hdrName, hdr

dataStore = '/Volume2/arunFiles/python_HSITools/crismBalancingDatasets/dataProducts/store_Composite_balancedDataset_wclassLabels.h5'
tableName = 'IF_mixedSamples'

# -------------------------------------------------------------------------------------------------------------------- #
# MODEL-1
# Generator : 6 Convolutional Layers
# Discriminator: 5 Convolutional Layers
# Filter Size:11

obj1 = GANModel_1d(img_rows=240, dropout=0.3, genFilters=500, disFilters=20, filterSize=11)
gen1 = obj1.genModel_CV_L6s2()
gen1.load_weights('/Volume2/arunFiles/python_HSITools/trainedModels/balancedDataSets/JSDGan/Model-1_big_5e-5_1Jan/generator/gen_cR_70.h5')
dis1 = obj1.disModel_CV_L6s2_old()
dis1.load_weights('/Volume2/arunFiles/python_HSITools/trainedModels/balancedDataSets/JSDGan/Model-1_big_5e-5_1Jan/discriminator/dis_cR_70.h5')


# Learning Rate = 2e-5, decay=0.5
#location = '/Volume2/arunFiles/python_HSITools/trainedModels/balancedDataSets/Models/Model-1_big_1e-4_24Dec/'
#objTrain1 = GANModel_Train(gen1, dis1, location, tableName, dataStore=dataStore, learningRate=1.e-4, verbose=True,
#                           noiseSize=50, batch_size=256, epochs=76, decayRate=0.5)
#objTrain1.GAN_trainFromStore()

# -------------------------------------------------------------------------------------------------------------------- #
# MODEL-2
# Generator : 6 Convolutional Layers
# Discriminator: 5 Convolutional Layers
# Filter Size:11

# Learning Rate = 2e-5, decay=0.5
location = '/Volume2/arunFiles/python_HSITools/trainedModels/balancedDataSets/JSDGan/Model-1_big_1e-5_2Jan/'
objTrain1 = GANModel_Train(gen1, dis1, location, tableName, dataStore=dataStore, learningRate=1 .e-5, verbose=True,
                           noiseSize=50, batch_size=256, epochs=76, decayRate=0.5)
objTrain1.GAN_trainFromStore()