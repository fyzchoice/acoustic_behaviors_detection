

class init_args:

    def __init__(self,Emg_feature_len=40,Emg_pad_len=320,audio_feature_len=152,audio_pad_len=45):
        self.Emg_feature_len=Emg_feature_len
        self.Emg_pad_len=Emg_pad_len
        self.audio_feature_len=audio_feature_len
        self.audio_pad_len=audio_pad_len

    def print_detail(self,flag=0):
        if(flag==0):
            print(self.Emg_feature_len,self.Emg_pad_len)
        else:
            print(self.audio_feature_len,self.audio_pad_len)

    def getargs(self,flag=0):
        if(flag==0):
            return self.Emg_feature_len,self.Emg_pad_len
        else:
            return self.audio_feature_len,self.audio_pad_len
    def setargs(self,pad_len=320):
        self.Emg_pad_len=pad_len




