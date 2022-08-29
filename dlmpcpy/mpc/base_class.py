from time import time
'''
    Some common functionalities
'''

class Base:
    baseprint = print
    def setSilent(slient=False):
        if slient:
            Base.baseprint = lambda s:None
            #Base.measure = lambda s:None
            Base.report = lambda s,x:None
        else:
            Base.baseprint = print
            #Base.measure = Base.doMeasure
            Base.report = Base.doReportRuntimeAndRestart

    def __init__(self):
        self._run_time = 0
        self._name = '--'

    def setName(self,name):
        self._name = name

    def getName(self):
        return self._name

    def info(self, info_message):
        Base.baseprint('[%s][INFO] %s' % (self._name, info_message) )

    def error(self, error_message):
        Base.baseprint('[%s][ERROR] %s' % (self._name, error_message) )
        quit()

    def doMeasure(self):
        self._run_time = time()
    measure = doMeasure

    def startMeasuring(self):
        Base.measure(self)

    def reportRuntimeAndRestart(self, report_string):
        Base.report(self, report_string)

    def returnRuntimeAndRestart(self):
        ret = time() - self._run_time
        Base.measure(self)
        return ret

    def doReportRuntimeAndRestart(self, report_string):
        Base.baseprint("[%s] " % self._name + report_string % (time() - self._run_time))
        Base.measure(self)
    report = doReportRuntimeAndRestart