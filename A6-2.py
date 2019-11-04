import glob
from shutil import copyfile

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
participants = glob.glob("assets//source_emotion//*")
for participant in participants:
    part = "%s" %participant[-4:] #participant number

    for sessions in glob.glob("%s\\*" %participant):
        for files in glob.glob("%s\\*" %sessions):
            current_session = files[28:-30]
            file = open(files, 'r')
            emotion = int(float(file.readline()))
            sourcefile_emotion = glob.glob("assets//source_images//%s//%s//*" %(part, current_session))[-1] #image with the emotion
            sourcefile_neutral = glob.glob("assets//source_images//%s//%s//*" %(part, current_session))[0] #image with neutral face
            path_neutral = "assets//sorted_set\\neutral\\%s" %sourcefile_neutral[32:]
            path_emotion = "assets//sorted_set\\%s\\%s" %(emotions[emotion], sourcefile_emotion[32:])
            copyfile(sourcefile_neutral, path_neutral)
            copyfile(sourcefile_emotion, path_emotion)