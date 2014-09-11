#-------------------------------------------------
#
# Project created by QtCreator 2014-02-24T11:09:36
# created By Somnath Mukherjee  Used for Aerial Image Processing Software



#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestOPencv
TEMPLATE = app

INCLUDEPATH += /opt/local/include
SOURCES += main.cpp\
        testopencv.cpp

HEADERS  += testopencv.h

FORMS    += testopencv.ui



LIBS  += -L/opt/local/lib


LIBS += -lopencv_core.2.4.7 -lopencv_highgui.2.4.7 -lopencv_imgproc.2.4.7 -lopencv_ml.2.4.7 -lopencv_stitching.2.4.7 -lopencv_features2d.2.4.7 -lopencv_objdetect.2.4.7  -lopencv_video.2.4.7 -v
LIBS += `pkg-config opencv --cflags --libs`
       #-lopencv_contrib.2.4.7 \
      # -lopencv_photo.2.4.7 \
      # -lopencv_features2d.2.4.7 \
       #-lopencv_objdetect.2.4.7 \
      # -lopencv_gpu.2.4.7 \
       #-lopencv_calib3d.2.4.7 \
       #-lopencv_video.2.4.7 \
       #-lopencv_flann.2.4.7
       #-lopencv_nonfree.2.4.7 -v

#LIBS += -lopencv_calib3d.2.4.7 -lopencv_video.2.4.7 -lopencv_flann.2.4.7 -lopencv_nonfree.2.4.7  -v
#LIBS+= -lopencv_legacy.2.4.7  -lopencv_videostab.2.4.7  -lopencv_superres.2.4.7  -v


