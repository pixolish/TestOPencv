
/* created By Somnath Mukherjee
Used for Aerial Image Processing Software
*/


#ifndef TESTOPENCV_H
#define TESTOPENCV_H

#include <QMainWindow>
//#include <QMainWindow>
#include <QLabel>
#include <QTimer>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace Ui {
class TestOpencv;
}

class TestOpencv : public QMainWindow
{
    Q_OBJECT

public:
    explicit TestOpencv(QWidget *parent = 0);
    ~TestOpencv();

private slots:
    void openImage();
    void toGrayscaleImg();
    void process();
    void process1();



    void process2();
    void process3();
    void processvideo(char* filename);
    IplImage  *show_histogram( IplImage* src, char* channel_name);

    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

    void on_pushButton_5_clicked();

    void on_pushButton_6_clicked();

    void on_pushButton_7_clicked();

    void on_progressBar_valueChanged(int value);

    //void on_pushButton_8_clicked(bool checked);

    void on_pushButton_8_clicked();
    void update();
    void updatepicture();



    void on_pushButton_9_clicked();

    void on_pushButton_10_clicked();

    void on_verticalScrollBar_actionTriggered(int action);
    IplImage* retrvframe(CvCapture *capture, int num);

private:
    Ui::TestOpencv *ui;

    QString fileName,filevideo;
    IplImage *iplImg;
    IplImage* pano;
    cv::Mat pan;
    cv::Mat img1;

    char *charFileName,*charfilevideo;
    QImage qimgNew;
    QImage qimgGray;
    QImage qimg;
    QImage qimg1;
    QImage qimg2;
    QImage qimg3;
    QImage qpano,red,green,blue;

    cv::vector<cv::Mat>imgarr;

    QTimer *timer;
    QTimer *timer1;
        unsigned int cntr;
    int global_c;
    CvCapture *cap, *cap1;
    int totalfrmnum;

   // QLabel lblImage;


};

#endif // TESTOPENCV_H
