import s3fs 

#grab filesystem 
fs = s3fs.S3FileSystem(anon=True)

#example grabbing GOES

files = fs.ls('s3://noaa-goes16/ABI-L1b-RadF/2018/347/20')

#lets grab all C14 images 
res = [i for i in files if 'OR_ABI-L1b-RadF-M6C14_G16' in i]
res.sort()

#
savedir = 'PATH'
for i in tqdm.tqdm(range(len(res))):
    fs.get(res[i], savedir+res[i].split('/')[-1])