import zipfile
import io
import numpy as np
from tifffile import TiffFile
from pyspark import SparkContext, SparkConf
import hashlib
from scipy import linalg #linalg contains a method svd
#from skimage import data  #from the “scikit-image” package

def getOrthoTif(zfBytes):
 #given a zipfile as bytes (i.e. from reading from a binary file),
 # return a np array of rgbx values for each pixel
 bytesio = io.BytesIO(zfBytes)
 zfiles = zipfile.ZipFile(bytesio, "r")
 #find tif:
 for fn in zfiles.namelist():
  if fn[-4:] == '.tif':#found it, turn into array:
   tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
   return tif.asarray()

def split_tiff(x,t):
#x should be a yXmXn matrix, and t should even divides m,n
#returns a list of 3D blocks of size yXtXt
    down =  range(0,x[1].shape[0],t)
    across = range(0,x[1].shape[1],t)
    reshaped = []
    for d in down:
        for a in across:
            reshaped.append(x[1][d:d+t,a:a+t,:])
    return reshaped

def reduce_resolution(x,t): #reduce by a factor of t, say t=10
    dims = x[1].shape
    k = int(dims[0]/t) #reduced matrix dimensions 50x50
    name = x[0]
    reshaped = np.zeros(shape=(k,k),dtype=float)
    array = x[1]
    for i in range(0,dims[0],t):
        for j in range(0,dims[1],t):
            ival = i//t
            jval = j//t
            reshaped[ival][jval] = np.mean(array[i:i+t,j:j+t])
    return (name,reshaped)

def append_name_to_subtiff(a):
    rem = a[1]%25
    ans = a[1]//25
    final_name = zf.value[ans]+'-'+str(rem)
    return (final_name,a[0])

def printAns1e(a):

    if a[0]=="3677454_2025195.zip-0" or a[0]=="3677454_2025195.zip-1" or a[0]=="3677454_2025195.zip-18" or a[0]=="3677454_2025195.zip-19":
        print('Answer for 1e:',a[0],a[1][0][0])
    return a

def printAns2f(a):
    if a[0]=="3677454_2025195.zip-1" or a[0]=="3677454_2025195.zip-18":
        print('Answer for 2f:',a)
    return a

def convertToIntensity(a):
    dims = a[1].shape
    intensity = np.zeros(shape=(dims[0],dims[1]),dtype=int)
    array = a[1]
    name = a[0]
    for i in range(500):
        for j in range(500):
            intensity[i][j] = int(sum(array[i][j][:-1])*array[i][j][-1]/300)
    return (name,intensity)

def row_diff(x):
    name = x[0]
    array = x[1]
    diff_array = np.diff(array)
    dims = diff_array.shape
    ans = np.zeros(shape=(dims[0],dims[1]),dtype=int)
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            if diff_array[i][j] < -1:
                ans[i][j] = -1
            elif diff_array[i][j] > 1:
                ans[i][j] = 1
            else:
                ans[i][j] = 0
    return name,ans

def col_diff(x):
    name = x[0]
    array = x[1]
    transpose_array = np.transpose(array)
    intermediate_diff_array = np.diff(transpose_array)
    diff_array = np.transpose(intermediate_diff_array)
    dims = diff_array.shape
    ans = np.zeros(shape=(dims[0],dims[1]),dtype=int)
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            if diff_array[i][j] < -1:
                ans[i][j] = -1
            elif diff_array[i][j] > 1:
                ans[i][j] = 1
            else:
                ans[i][j] = 0
    return name,ans

def send_to_bins(a,bins,bands):
    name = a[0]
    array = a[1]
    bin_alloted = np.zeros(shape=bands,dtype=int)
    cnt = 0
    step = 128//bands
    for i in range(0,len(array),step):
        sum = 0
        for j in range(0,step):
            sum = sum+array[i+j]
        bin_alloted[cnt] = sum%bins
        cnt = cnt+1
    return (name,bin_alloted)

def hashFunc(a):
    array = a[1] #4900 vector
    name = a[0]
    #Chunking logic
    feat_len = len(array)
    bin_size1 = feat_len//128
    excess = feat_len - (bin_size1*128) #Chunking logic : binsize2 - excess times, binsize1 - 128-excess times
    bin_size2 = bin_size1+1
    bin_number1 = 128-excess
    bin_number2 = excess
    chunks = []
    cnt = 0
    for i in range(0,bin_number1):
        chunks.append(array[cnt:cnt+bin_size1])
        cnt = cnt+bin_size1
    for i in range(0,bin_number2):
        chunks.append(array[cnt:cnt+bin_size2])
        cnt = cnt+bin_size2
    hash_signature = np.zeros(shape=(128),dtype=int)
    cnt = 0
    hashId = hashlib.md5()
    for j in range(0,128):
        hashId.update(repr(chunks[j]).encode('utf-8'))
        message_digest = hashId.hexdigest()
        number = int(message_digest, 16)
        mod_of_digits = sum(int(digit) for digit in str(number)) % 10
        hash_signature[cnt] = mod_of_digits
        cnt = cnt+1
    return (name,hash_signature)


def compute_svd(x):
    TwoDArray = []
    NameList = []
    while True:
        try:
            a = next(x)
            name = a[0]
            NameList.append(name)
            array = a[1]
            TwoDArray.append(array)
        except StopIteration:
            print("No rows")
            break
    TwoDArray = np.matrix(TwoDArray)
    U, s, Vh = linalg.svd(TwoDArray, full_matrices=1)
    low_dim_p = 10
    TwoDArray_lowdim = U[:, 0:low_dim_p]
    Ans = []
    for i in range(len(NameList)):
        Ans.append([NameList[i],TwoDArray_lowdim[i]])
    return Ans

if __name__ == "__main__":
    sc = SparkContext('local[2]')
    file_path = 'C:\\SBU\\Fall2017\\BigData\\Assignment2\\large_sample'

    rdd = sc.binaryFiles(file_path)
    #For resolution 10
    resolution = 10

    result_1e = rdd.map(lambda x: (x[0].split('/')[-1],getOrthoTif(x[1]))).map(lambda x: (x[0],split_tiff(x,500)))\
        .flatMap(lambda x: [(x[0]+'-'+str(i),x[1][i]) for i in range(len(x[1]))]).map(lambda x:printAns1e(x)).map(lambda x:convertToIntensity(x))

    result_2b = result_1e.map(lambda x:reduce_resolution(x,resolution)).persist()

    rowdiffrdd = result_2b.map(lambda x:row_diff(x)).map(lambda x:(x[0],x[1].flatten()))
    coldiffrdd = result_2b.map(lambda x:col_diff(x)).map(lambda x:(x[0],x[1].flatten()))

    bin_size = 600
    band_size = 2
    feature_rdd = rowdiffrdd.union(coldiffrdd).reduceByKey(lambda x,y : np.concatenate([x,y])).map(lambda x:printAns2f(x)).persist()
    chunks_rdd = feature_rdd.map(lambda x: hashFunc(x))

    bin_array = chunks_rdd.map(lambda x:send_to_bins(x,bin_size,band_size)).collect()

    comp_files = ['3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19']
    ans_3b = {}
    input_dict = {}

    for file in comp_files:
        for i in bin_array:
            if file==i[0]:
                input_dict[file] = i[1]
                ans_3b[file] = []

    for file in comp_files:
        for i in bin_array:
            if file != i[0]:
                for j in range(len(i[1])):
                    if i[1][j] == input_dict[file][j]: #compute similarity
                        ans_3b[file].append(i[0])
    print("==========================================3b===============================================")
    print("Lengths:")
    print('3677454_2025195.zip-0',len(ans_3b['3677454_2025195.zip-0']))
    print('3677454_2025195.zip-1',len(ans_3b['3677454_2025195.zip-1']))
    print('3677454_2025195.zip-18',len(ans_3b['3677454_2025195.zip-18']))
    print('3677454_2025195.zip-19',len(ans_3b['3677454_2025195.zip-19']))
    print("Similar images for 3677454_2025195.zip-1")
    print('3677454_2025195.zip-1',ans_3b['3677454_2025195.zip-1'])
    print("Similar images for 3677454_2025195.zip-18")
    print('3677454_2025195.zip-18',ans_3b['3677454_2025195.zip-18'])
    svd = feature_rdd.partitionBy(10).mapPartitions(lambda x: compute_svd(x)).collect()

    for i in range(len(svd)):
        if(svd[i][0] == '3677454_2025195.zip-1'):
            vector_zip1 = svd[i][1]
        if(svd[i][0] == '3677454_2025195.zip-18'):
            vector_zip18 = svd[i][1]
    print("3c========Zip 1=======================================================")
    ans_3c1 = []
    for i in range(len(ans_3b['3677454_2025195.zip-1'])):
        for j in range(len(svd)):
            if(ans_3b['3677454_2025195.zip-1'][i]==svd[j][0]):
                similar_dist = np.linalg.norm(svd[j][1]-vector_zip1)
                ans_3c1.append([ans_3b['3677454_2025195.zip-1'][i],similar_dist])

    ans_3c1.sort(key=lambda tup: tup[1])
    print("Euclidean distances for 3677454_2025195.zip-1")
    print(ans_3c1)
    print("3c=========Zip 18=======================================================")
    ans_3c2 = []
    for i in range(len(ans_3b['3677454_2025195.zip-18'])):
        for j in range(len(svd)):
            if (ans_3b['3677454_2025195.zip-18'][i] == svd[j][0]):
                similar_dist = np.linalg.norm(svd[j][1] - vector_zip18)
                ans_3c2.append([ans_3b['3677454_2025195.zip-18'][i], similar_dist])

    ans_3c2.sort(key=lambda tup: tup[1])
    print("Euclidean distances for 3677454_2025195.zip-18")
    print(ans_3c2)

    # resolution = 5
    #
    # result_1e = rdd.map(lambda x: (x[0].split('/')[-1], getOrthoTif(x[1]))).map(lambda x: (x[0], split_tiff(x, 500))) \
    #     .flatMap(lambda x: [(x[0] + '-' + str(i), x[1][i]) for i in range(len(x[1]))]).map(lambda x: printAns1e(x)).map(
    #     lambda x: convertToIntensity(x))
    #
    # result_2b = result_1e.map(lambda x: reduce_resolution(x, resolution)).persist()
    #
    # rowdiffrdd = result_2b.map(lambda x: row_diff(x)).map(lambda x: (x[0], x[1].flatten()))
    # coldiffrdd = result_2b.map(lambda x: col_diff(x)).map(lambda x: (x[0], x[1].flatten()))
    #
    # bin_size = 1700
    # band_size = 2
    # feature_rdd = rowdiffrdd.union(coldiffrdd).reduceByKey(lambda x, y: np.concatenate([x, y])).map(
    #     lambda x: printAns2f(x)).persist()
    # chunks_rdd = feature_rdd.map(lambda x: hashFunc(x))
    #
    # bin_array = chunks_rdd.map(lambda x: send_to_bins(x, bin_size, band_size)).collect()
    #
    # comp_files = ['3677454_2025195.zip-0', '3677454_2025195.zip-1', '3677454_2025195.zip-18', '3677454_2025195.zip-19']
    # ans_3b = {}
    # input_dict = {}
    #
    # for file in comp_files:
    #     for i in bin_array:
    #         if file == i[0]:
    #             input_dict[file] = i[1]
    #             ans_3b[file] = []
    #
    # for file in comp_files:
    #     for i in bin_array:
    #         if file != i[0]:
    #             for j in range(len(i[1])):
    #                 if i[1][j] == input_dict[file][j]:
    #                     ans_3b[file].append(i[0])
    # print("==========================================3b===============================================")
    #
    # print('3677454_2025195.zip-0', len(ans_3b['3677454_2025195.zip-0']))
    # print('3677454_2025195.zip-1', len(ans_3b['3677454_2025195.zip-1']), ans_3b['3677454_2025195.zip-1'])
    # print('3677454_2025195.zip-18', len(ans_3b['3677454_2025195.zip-18']), ans_3b['3677454_2025195.zip-18'])
    # print('3677454_2025195.zip-19', len(ans_3b['3677454_2025195.zip-19']))
    #
    # svd = feature_rdd.partitionBy(10).mapPartitions(lambda x: compute_svd(x)).collect()
    # # print("Checkkkkkkkkkkkkkk",svd)
    # for i in range(len(svd)):
    #     if (svd[i][0] == '3677454_2025195.zip-1'):
    #         vector_zip1 = svd[i][1]
    #     if (svd[i][0] == '3677454_2025195.zip-18'):
    #         vector_zip18 = svd[i][1]
    # print("Zip 1=======================================================")
    # ans_3c1 = []
    # for i in range(len(ans_3b['3677454_2025195.zip-1'])):
    #     for j in range(len(svd)):
    #         if (ans_3b['3677454_2025195.zip-1'][i] == svd[j][0]):
    #             similar_dist = np.linalg.norm(svd[j][1] - vector_zip1)
    #             ans_3c1.append([ans_3b['3677454_2025195.zip-1'][i], similar_dist])
    #
    # ans_3c1.sort(key=lambda tup: tup[1])
    # print(ans_3c1)
    # print("Zip 18=======================================================")
    # ans_3c2 = []
    # for i in range(len(ans_3b['3677454_2025195.zip-18'])):
    #     for j in range(len(svd)):
    #         if (ans_3b['3677454_2025195.zip-18'][i] == svd[j][0]):
    #             similar_dist = np.linalg.norm(svd[j][1] - vector_zip18)
    #             ans_3c2.append([ans_3b['3677454_2025195.zip-18'][i], similar_dist])
    #
    # ans_3c2.sort(key=lambda tup: tup[1])
    # print(ans_3c2)
    #
    #
    #
