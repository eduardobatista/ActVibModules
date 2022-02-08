import pandas as pd

class ActVibData:

    def __init__(self,filename):
        self.filename = filename
        self.data = pd.read_feather(filename)
        if "Log" in self.data.columns:
            self.hasLog = True
        else:
            self.hasLog = False

    def getTime(self):
        return self.data["Tempo (s)"].values

    def getSignalNames(self):
        return list(self.data.columns)

    def getSignal(self,signalname):
        """
            Signal name can be:
               Tempo (s),
               IMU1AccX, IMU1AccY, IMU1AccZ, 
               IMU1GyroX,IMU1GyroY, IMU1GyroZ, 
               IMU2AccX, IMU2AccY, IMU2AccZ, 
               IMU2GyroX, IMU2GyroY, IMU2GyroZ,
               IMU3AccX, IMU3AccY, IMU3AccZ, 
               IMU3GyroX, IMU3GyroY, IMU3GyroZ 
        """
        return self.data[signalname].values
    
    def getAccX(self,imuidx=1):
        return self.data[f"IMU{imuidx}AccX"].values
    
    def getAccY(self,imuidx=1):
        return self.data[f"IMU{imuidx}AccY"].values

    def getAccZ(self,imuidx=1):
        return self.data[f"IMU{imuidx}AccZ"].values

    def getGyroX(self,imuidx=1):
        return self.data[f"IMU{imuidx}GyroX"].values
    
    def getGyroY(self,imuidx=1):
        return self.data[f"IMU{imuidx}GyroY"].values

    def getGyroZ(self,imuidx=1):
        return self.data[f"IMU{imuidx}GyroZ"].values

    def getDACData(self,dacid=1):
        if (dacid < 1) or (dacid > 4):
            raise BaseException("DACid must be between 1 and 4.")
        daccols = list(filter(lambda x: x.startswith("DAC"), self.getSignalNames()))
        return self.data[daccols[dacid-1]].values

    def getDAC1kHzData(self):
        daccols = list(filter(lambda x: x.startswith("DAC"), self.getSignalNames()))
        return self.data[daccols].values.reshape((self.data.shape[0]*4))

    def getNotes(self):
        if not self.hasLog:
            raise BaseException(f"No notes found in {self.filename}.")
        notes = self.data["Log"].head(1).values.tolist()[0]
        if notes == "Started":
            raise BaseException(f"No notes found in {self.filename}.")
        return notes

    def getLogs(self):
        if not self.hasLog:
            raise BaseException(f"No logs found in {self.filename}.")
        logs = self.data[["Tempo (s)","Log"]][self.data["Log"].notnull()].values.tolist()
        if logs[0][1] != "Started":
            logs = logs[1:]
        return logs
        