/* created By Somnath Mukherjee
Used for Aerial Image Processing Software
*/




#include "testopencv.h"
#include "ui_testopencv.h"
#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QLabel>
#include <stdio.h>
#include <QMessageBox>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavresample/avresample.h>
#include <libavfilter/avfilter.h>
int GRAYLEVEL = 256;
#define MAX_BRIGHTNESS 255

//#include <>

bool chk=true;
int count =0;
TestOpencv::TestOpencv(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::TestOpencv)
{
    ui->setupUi(this);
    ui->pushButton_8->setEnabled(false);
    // timer->stop();
    //ui->setupUi    );
       /* timer = new QTimer();
        cntr = 0;
        connect(timer, SIGNAL(timeout()), this, SLOT( update() ) );*/
}

TestOpencv::~TestOpencv()
{
    delete ui;
}
IplImage* TestOpencv::show_histogram ( IplImage *src, char *channel_name)
{
    IplImage* img, *canvas;



      int       bins = 256;
      int       hist[bins];
      double    scale;
      int       i, j, channel, max = 0;

      CvScalar   colors[] = { CV_RGB(0,0,255), CV_RGB(0,255,0),
                              CV_RGB(255,0,0), CV_RGB(0,0,0) };

      channel = strcmp(channel_name, "blue")  == 0 ? 0
              : strcmp(channel_name, "green") == 0 ? 1
              : strcmp(channel_name, "red")   == 0 ? 2
              : strcmp(channel_name, "gray")  == 0 ? 3 : 0;

      if (src->nChannels == 3 && channel == 3)
      {
        img = cvCreateImage(cvGetSize(src), 8, 1);
        cvCvtColor(src, img, CV_BGR2GRAY);
      }
      else if (channel > src->nChannels)
        exit;
      else
        img = cvCloneImage(src);

      canvas = cvCreateImage(cvSize(256, 125), IPL_DEPTH_8U, 3);
      cvSet(canvas, CV_RGB(255,255,255), NULL);

      /* Reset histogram */
      for (j = 0; j < bins-1; hist[j]=0, j++);

      /* Calc histogram of the image */
      for (i = 0; i < img->height; i++)
      {
        uchar* ptr = (uchar*)(img->imageData + i * img->widthStep);
        for (j = 0; j < img->width; j+=img->nChannels)
          hist[ptr[j+(channel == 3 ? 0 : channel)]]++;
      }

      /* Get histogram peak */
      for (i = 0; i < bins-1; i++)
        max = hist[i] > max ? hist[i] : max;

      /* Get scale so the histogram fit the canvas height */
      scale = max > canvas->height ? (double)canvas->height/max : 1.;

      /* Draw histogram */
      for (i = 0; i < bins-1; i++)
      {
        CvPoint pt1 = cvPoint(i, canvas->height - (hist[i] * scale));
        CvPoint pt2 = cvPoint(i, canvas->height);
        cvLine(canvas, pt1, pt2, colors[channel], 1, 8, 0);
      }

      cvReleaseImage(&img);

      return canvas;



}

IplImage* binarize_otsu(IplImage* image)

{

    IplImage* imgBin =cvCreateImage(cvGetSize(image),8,1);
    float  hist[GRAYLEVEL];
    double prob[GRAYLEVEL],omega[GRAYLEVEL];  // prob of graylevels
    double myu[GRAYLEVEL];    // mean value for separation
    double max_sigma,sigma[GRAYLEVEL];  // inter-class variance

    int i, x, y; /* Loop variable */
    int threshold; /* threshold for binarization */

    // Histogram generation
    memset((float*)   hist , 0, GRAYLEVEL * sizeof(float)   );

    CvSize size = cvGetSize(image);

    for (int i = 0; i < size.height; ++i)
    {
        unsigned char* pData = (unsigned char*) (image->imageData + i *
                                                 image->widthStep);
        for (int j = 0; j < size.width; ++j)
        {
            int k = (float)((unsigned char) *(pData+j));
            hist[k]++;
        }
    }

    int taille = size.width * size.height;

    // calculation of probability density
    for ( i = 0; i < GRAYLEVEL; ++i )
    {
        prob[i] = (double) ((double)hist[i] / (double)taille);
    }


    // omega & myu generation
    omega[0] = prob[0];
    myu[0]   = 0.0;
    for (i = 1; i < GRAYLEVEL; i++)
    {
        omega[i] = omega[i-1] + prob[i];
        myu[i]   = myu[i-1]   + (i*prob[i]);
    }

    //-----------------------------------------------------------------
    // sigma maximization
    //  sigma stands for inter-class variance
    //  and determines optimal threshold value
    //----------------------------------------------------------------
    threshold = 0;
    max_sigma = 0.0;
    for (i = 0; i < GRAYLEVEL-1; i++)
    {
        if (omega[i] != 0.0 && omega[i] != 1.0)
        {
            //sigma[i] =  (omega[i]*(1.0 - omega[i])) * ((myu[GRAYLEVEL-1] - 2*myu[i]) *
            (myu[GRAYLEVEL-1] - 2*myu[i]);
            sigma[i] = ((myu[GRAYLEVEL-1]*omega[i] - myu[i]) *
                        (myu[GRAYLEVEL-1]*omega[i] - myu[i])) /  (omega[i]*(1.0 - omega[i]));
        }
        else
        {
            sigma[i] = 0.0;
        }
        if (sigma[i] > max_sigma)
        {
            max_sigma = sigma[i];
            threshold = i;
        }
    }

    printf("threshold = %d\n", threshold);

    // binarization output into imgBin
    for (y = 0; y < size.height; ++y)
    {
        unsigned char* pData    = (unsigned char*) (image->imageData  + (y *
                                                                         image->widthStep));
        unsigned char* pDataBin = (unsigned char*) (imgBin->imageData + (y *
                                                                         imgBin->widthStep));
        for (x = 0; x < size.width; ++x)
        {
            if ( *(pData+x) > threshold)
            {
                *(pDataBin+x) = threshold;
            }
            else
            {
                *(pDataBin+x) = 0;
            }
        }
    }



    return imgBin;


}


void TestOpencv::processvideo(char* filename)
{
    CvCapture *cap =cvCaptureFromFile(filename);
    printf("Entering Video");
    if(!cap)
        exit;
    printf("Entering Video");
    int frmcnt =(int)cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_COUNT);
    int i=0;
    while (i<frmcnt)
    {

        IplImage* img =cvQueryFrame(cap);

        QImage qimgvid = QImage((const unsigned char*)img->imageData,img->width,img->height,QImage::Format_RGB888).rgbSwapped();
        ui->label->setPixmap(QPixmap::fromImage(qimgvid));
        cvWaitKey(100);
        cvReleaseImage(&img);

       i=i+1;
    }

   cvReleaseCapture(&cap);


}

void TestOpencv::openImage()
{


    ui->label->setText("Aerial Image Processing");

    fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::currentPath(),tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png ,*.pgm, *.gif]"));
    charFileName = fileName.toLocal8Bit().data();
    iplImg = cvLoadImage(charFileName);
    cvShowImage("ImageView",iplImg);

    qimgNew = QImage((const unsigned char*)iplImg->imageData,iplImg->width,iplImg->height,QImage::Format_RGB888).rgbSwapped();
    ui->label->setPixmap(QPixmap::fromImage(qimgNew));

    //cvResize(iplImg,iplImg,3);
    qimg1.load(charFileName);
    QImage img =qimgNew.scaled(220,180,Qt::IgnoreAspectRatio);

    ui->label_2->setPixmap(QPixmap::fromImage(img));


}
void TestOpencv::toGrayscaleImg()
{
    //ui->label->clear();
    IplImage *imgGray=cvCreateImage(cvGetSize(iplImg),8,1); ;
    cvCvtColor(iplImg,imgGray,CV_BGR2GRAY);
    //= cvLoadImage(charFileName, CV_LOAD_IMAGE_GRAYSCALE);
    IplImage *imgrgb=cvCloneImage(iplImg); //= cvLoadImage(charFileName);
    qimgGray = QImage((const unsigned char*)imgGray->imageData,imgGray->width,imgGray->height,QImage::Format_Indexed8);
    qimgGray.setPixel(0,0,qRgb(0,0,0));
    ui->label->setPixmap(QPixmap::fromImage(qimgGray));
    printf("---Executing---2");

    IplImage *red1 = show_histogram(imgrgb,"red");
    IplImage *green1 =show_histogram(imgrgb,"green");
    IplImage *blue1 =show_histogram(imgrgb,"blue");
    red =QImage((const unsigned char*)red1->imageData,red1->width,red1->height,QImage::Format_RGB888).rgbSwapped();
    red.scaled(180,70,Qt::IgnoreAspectRatio);
    ui->red->setPixmap(QPixmap::fromImage(red));
    green =QImage((const unsigned char*)green1->imageData,green1->width,green1->height,QImage::Format_RGB888).rgbSwapped();
    green.scaled(180,70,Qt::IgnoreAspectRatio);
    ui->green->setPixmap(QPixmap::fromImage(green));
    blue =QImage((const unsigned char*)blue1->imageData,blue1->width,blue1->height,QImage::Format_RGB888).rgbSwapped();
    blue.scaled(180,70,Qt::IgnoreAspectRatio);
    ui->blue->setPixmap(QPixmap::fromImage(blue));

    IplImage *hist =binarize_otsu(imgGray);

    cvNamedWindow("Binarized");
    cvShowImage("Binarized",hist);




}

void TestOpencv::process()
{



    IplImage *imgray=cvCreateImage(cvGetSize(iplImg),8,1); ;
    cvCvtColor(iplImg,imgray,CV_BGR2GRAY);
    //= cvLoadImage(charFileName, CV_LOAD_IMAGE_GRAYSCALE);
    IplImage *imgrgb=cvCloneImage(iplImg);
    //
    //cvNamedWindow("window",1);


    IplImage *img =cvCreateImage(cvGetSize(imgray),IPL_DEPTH_8U,3);
   //IplImage *img1;// =cvCreateImage(cvSize(imgray->width,imgray->height),8,1);
    IplImage *img1 =cvCreateImage(cvGetSize(imgray),IPL_DEPTH_8U,1);


    printf("\n---Its Running-----\n");
    cvThreshold(imgray,img1,0,255,CV_THRESH_OTSU);

    //cvOr(imgray,img1,img1,NULL);
//cvShowImage("window",img1);
    qimg = QImage((const unsigned char*)img1->imageData,img1->width,img1->height,QImage::Format_Indexed8);
    qimg.setPixel(0,0,qRgb(0,0,0));
    ui->label->setPixmap(QPixmap::fromImage(qimg));



}

void TestOpencv::process1()
{
    IplImage *imgray=cvCreateImage(cvGetSize(iplImg),8,1); ;
    cvCvtColor(iplImg,imgray,CV_BGR2GRAY);
    //= cvLoadImage(charFileName, CV_LOAD_IMAGE_GRAYSCALE);
    IplImage *imgrgb=cvCloneImage(iplImg);
    //
    //cvNamedWindow("window",1);


    IplImage *img =cvCreateImage(cvGetSize(imgray),IPL_DEPTH_8U,3);
   //IplImage *img1;// =cvCreateImage(cvSize(imgray->width,imgray->height),8,1);
    IplImage *img1 =cvCreateImage(cvGetSize(imgray),IPL_DEPTH_8U,1);


    printf("\n---Its Running-----\n");

    IplImage *hsl,*h,*s,*l,*m;
    hsl=cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
    h=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
    s=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
    l=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
    m=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
    cvCvtColor(imgrgb,hsl,CV_RGB2HLS);
    IplImage* totalimg ;
    int ch[]= { 0,0, 1,0, 2,0 };

    //cvMixChannels(&hsl,1,totalimg,3,ch,3);

    cvSplit(hsl,h,s,l,NULL);
 //cvShowImage("window",imgrgb);
printf("Image Splitting");

    cvThreshold(h,img1,0,255,CV_THRESH_OTSU);

    //cvOr(imgray,img1,img1,NULL);
    for (int i=0;i<imgray->height;i++)
    {
        for(int j=0;j<imgray->width;j++)
        {
          CvScalar v,v1,v2;

          v=cvGet2D(imgrgb,i,j);
          v1=cvGet2D(img1,i,j);
          if(v1.val[0]==0)
          {
              v.val[0]=0;
              v.val[1]=0;
              v.val[2]=0;
          }
          else
          {
              v.val[0]=v.val[0];
              v.val[1]=v.val[1];
              v.val[2]=v.val[2];
          }

          cvSet2D(imgrgb,i,j,v);

        }
    }


//cvShowImage("window",img1);
    qimg2 = QImage((const unsigned char*)imgrgb->imageData,imgrgb->width,imgrgb->height,QImage::Format_RGB888).rgbSwapped();
    //qimg.setPixel(0,0,qRgb(0,0,0));
    ui->label->setPixmap(QPixmap::fromImage(qimg2));
}

void TestOpencv::process2()
{
    IplImage *imgray=cvCreateImage(cvGetSize(iplImg),8,1); ;
    cvCvtColor(iplImg,imgray,CV_BGR2GRAY);
    //= cvLoadImage(charFileName, CV_LOAD_IMAGE_GRAYSCALE);
    IplImage *imgrgb=cvCloneImage(iplImg);
    //
    //cvNamedWindow("window",1);


    IplImage *img =cvCreateImage(cvGetSize(imgray),IPL_DEPTH_8U,3);
   //IplImage *img1;// =cvCreateImage(cvSize(imgray->width,imgray->height),8,1);
    IplImage *img1 =cvCreateImage(cvGetSize(imgray),IPL_DEPTH_8U,1);

    cvCanny(imgray,img1,140,180,3);

    for (int i=0;i<imgray->height;i++)
    {
        for(int j=0;j<imgray->width;j++)
        {
          CvScalar v,v1,v2;

          v=cvGet2D(imgrgb,i,j);
          v1=cvGet2D(img1,i,j);

              v.val[0]=v.val[0]+v1.val[0];
              v.val[1]=v.val[1]+v1.val[0];
              v.val[2]=v.val[2]+v1.val[0];

              if(v.val[0]>256)
              {
                 v.val[0] =0;//((256*256)/v.val[0]) ;
              }
              else if(v.val[1]>256)
              {
                  v.val[1] =255;//((256*256)/v.val[1]);
              }
              else if(v.val[2]>256)
              {
                  v.val[2] =255;//((256*256)/v.val[2]);
              }



          cvSet2D(imgrgb,i,j,v);

        }
    }


    qimg3 = QImage((const unsigned char*)imgrgb->imageData,imgrgb->width,imgrgb->height,QImage::Format_RGB888).rgbSwapped();
    //qimg.setPixel(0,0,qRgb(0,0,0));
    ui->label->setPixmap(QPixmap::fromImage(qimg3));



}

void TestOpencv::process3()
{

cv::Mat pan(800,450,CV_8UC3);
//timer->start(ui->progressBar->text().toInt() * 60000  / 60);



cv::Stitcher stitcher = cv::Stitcher::createDefault(false);


stitcher.setRegistrationResol(0.9);

stitcher.setSeamEstimationResol(0.1);

// stitcher.setCompositingResol(ORIG_RESOL);

stitcher.setPanoConfidenceThresh(0.9);

stitcher.setWaveCorrection(true);

stitcher.setWaveCorrectKind(cv::detail::WAVE_CORRECT_HORIZ);

stitcher.setFeaturesMatcher(new cv::detail::BestOf2NearestMatcher(false));

stitcher.setBundleAdjuster(new cv::detail::BundleAdjusterRay());







//stitcher.setPanoConfidenceThresh(0.8);

stitcher.setBlender( cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, false));



stitcher.setExposureCompensator (cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN_BLOCKS) );



cv::Stitcher::Status status = stitcher.stitch(imgarr, pan);

if (status != cv::Stitcher::OK)
    {
   printf("Can't stitch images, error code = %d",int(status) )  ;
        //return 0;
    }
//======Homography Estimation
/*
cv::namedWindow("panorama");
cv::Mat newresult,mapx,mapy,map;
    newresult.create(pan.size(),CV_32FC1);
    mapx.create(pan.size(),CV_32FC1);
    mapy.create(pan.size(),CV_32FC1);
    cv::Point2f points1[4];
    cv::Point2f points2[4];
    //remap( result1,newresult, mapx, mapy, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(255,0, 0) );
    points1[0].x=20;
    points1[0].y=20;

    points1[1].x=pan.cols-15;
    points1[1].y=14;

    points1[2].x=15;
    points1[2].y=pan.rows-15;

    points1[3].x=pan.cols;
    points1[3].y=pan.rows-20;

    points2[0].x=0;
    points2[0].y=0;

    points2[1].x=450;
    points2[1].y=0;

    points2[2].x=0;
    points2[2].y=300;

    points2[3].x=450;
    points2[3].y=300;
   // getperspec

    map =cv::getPerspectiveTransform(points1,points2);
    cv::warpPerspective(pan,pan,map,cv::Size(450,300));

*/


//=================

cv::imshow("panoroma",pan);
//pano =cvCloneImage(&(IplImage)pan);
qpano= QImage((const unsigned char*)pan.data,pan.cols,pan.rows,pan.step,QImage::Format_RGB888).rgbSwapped();
qpano.scaled(450,350,Qt::IgnoreAspectRatio);
ui->label->setPixmap(QPixmap::fromImage(qpano));
count =0;
//cvWaitKey();
cv::destroyWindow("panorama");
}

void TestOpencv::on_pushButton_clicked()
{
   openImage();
}

void TestOpencv::on_pushButton_2_clicked()
{
    printf("---Executing---1");
    toGrayscaleImg();
}

void TestOpencv::on_pushButton_3_clicked()
{
    ui->label->clear();
    ui->label_2->clear();
    ui->red->clear();
    ui->green->clear();
    ui->blue->clear();
    ui->pushButton_8->setEnabled(false);
    cv::destroyAllWindows();

}

void TestOpencv::on_pushButton_4_clicked()
{
  process();
}

void TestOpencv::on_pushButton_5_clicked()
{
    process1();
}

void TestOpencv::on_pushButton_6_clicked()
{
 process2();
}

void TestOpencv::on_pushButton_7_clicked()
{

   //chk=false;
    printf("----%d--\n",count);
    if(count<3)
    {
   QMessageBox::information(this,tr("The Title"),tr("Please add atleast more than two images for Panoramic View") );
    }
   //if(chk!=true)

     ui->pushButton_8->setEnabled(true);

     if(count >=3)
     {

   process3();
     }
  }

void TestOpencv::on_progressBar_valueChanged(int value)
{

    value =0;
    //value=value+5;
}

/*
void TestOpencv::on_pushButton_8_clicked(bool checked)
{
    fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::currentPath(),tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));


IplImage *img = cvLoadImage(charFileName);
    cv::Mat img1=cv::cvarrToMat(img);
    imgarr.push_back(img1);
    count =count +1;

}

*/

void TestOpencv::on_pushButton_8_clicked()
{


    fileName = QFileDialog::getOpenFileName(this,tr("Open Image"),QDir::currentPath(),tr("Image Files [ *.jpg , *.jpeg , *.bmp , *.png , *.gif]"));
    charFileName = fileName.toLocal8Bit().data();
    IplImage *img = cvLoadImage(charFileName);
    img1=cv::cvarrToMat(img);
    imgarr.push_back(img1);
    count =count +1;

}

void TestOpencv::update()
{
    cntr++;
    ui->progressBar->setValue( cntr ); //Should be incremented by one
    if( ui->progressBar->value() == 60 )
    {
        timer->stop();

        delete timer;
    }
}
void TestOpencv::updatepicture()
{


     IplImage* imgvideo =cvQueryFrame(cap);

     QImage qimgvid = QImage((const unsigned char*)imgvideo->imageData,imgvideo->width,imgvideo->height,QImage::Format_RGB888).rgbSwapped();
     qimgvid.scaled(550,450,Qt::IgnoreAspectRatio);
     ui->label->setPixmap(QPixmap::fromImage(qimgvid));


              IplImage *red1 = show_histogram(imgvideo,"red");
              IplImage *green1 =show_histogram(imgvideo,"green");
              IplImage *blue1 =show_histogram(imgvideo,"blue");
              red =QImage((const unsigned char*)red1->imageData,red1->width,red1->height,QImage::Format_RGB888).rgbSwapped();
              red.scaled(180,70,Qt::IgnoreAspectRatio);
              ui->red->setPixmap(QPixmap::fromImage(red));
              green =QImage((const unsigned char*)green1->imageData,green1->width,green1->height,QImage::Format_RGB888).rgbSwapped();
              green.scaled(180,70,Qt::IgnoreAspectRatio);
              ui->green->setPixmap(QPixmap::fromImage(green));
              blue =QImage((const unsigned char*)blue1->imageData,blue1->width,blue1->height,QImage::Format_RGB888).rgbSwapped();
              blue.scaled(180,70,Qt::IgnoreAspectRatio);
              ui->blue->setPixmap(QPixmap::fromImage(blue));



              ///==========Video Processing

              /*

              IplImage *imgray=cvCreateImage(cvGetSize(imgvideo),8,1); ;
              cvCvtColor(imgvideo,imgray,CV_BGR2GRAY);
              //= cvLoadImage(charFileName, CV_LOAD_IMAGE_GRAYSCALE);
              IplImage *imgrgb=cvCloneImage(imgvideo);
              //
              //cvNamedWindow("window",1);


              IplImage *img =cvCreateImage(cvGetSize(imgray),IPL_DEPTH_8U,3);
             //IplImage *img1;// =cvCreateImage(cvSize(imgray->width,imgray->height),8,1);
              IplImage *img1 =cvCreateImage(cvGetSize(imgray),IPL_DEPTH_8U,1);


              printf("\n---Its Running-----\n");

              IplImage *hsl,*h,*s,*l,*m;
              hsl=cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
              h=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
              s=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
              l=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
              m=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
              cvCvtColor(imgrgb,hsl,CV_RGB2HLS);
              IplImage* totalimg ;
              int ch[]= { 0,0, 1,0, 2,0 };

              //cvMixChannels(&hsl,1,totalimg,3,ch,3);

              cvSplit(hsl,h,s,l,NULL);
           //cvShowImage("window",imgrgb);
          printf("Image Splitting");

              cvThreshold(h,img1,0,255,CV_THRESH_OTSU);

              //cvOr(imgray,img1,img1,NULL);
              for (int i=0;i<imgray->height;i++)
              {
                  for(int j=0;j<imgray->width;j++)
                  {
                    CvScalar v,v1,v2;

                    v=cvGet2D(imgrgb,i,j);
                    v1=cvGet2D(img1,i,j);
                    if(v1.val[0]==0)
                    {
                        v.val[0]=0;
                        v.val[1]=0;
                        v.val[2]=0;
                    }
                    else
                    {
                        v.val[0]=v.val[0];
                        v.val[1]=v.val[1];
                        v.val[2]=v.val[2];
                    }

                    cvSet2D(imgrgb,i,j,v);

                  }
              }

        cvShowImage("Claasified",imgrgb);*/




     //ui->label->showFullScreen();
    //global_c ++;
   //printf("__Value----%d\n\n",global_c);

}

IplImage* TestOpencv::retrvframe(CvCapture *capture, int num)
{
    //if (!capture)
      //  exit();
    int count =0;
    IplImage *frame;

    while(count<num)
    {
      frame=cvQueryFrame(capture);
      count++;

    }
   return frame;
}

void TestOpencv::on_pushButton_9_clicked()
{


    QMessageBox::information(this,tr("The Title"),tr("Please Select An Aerial Video") );

   // filevideo = QFileDialog::getOpenFileName(this,tr("Operation6"),QDir::currentPath(),tr("Video Files [ *.mpeg , *.mp4 , *.vob , *.avi ,*.mov ]"));
   // CvCapture *cap =cvCaptureFromAVI(filevideo.toStdString().c_str());
    cap =cvCaptureFromFile("/Users/pranavagarwal/Desktop/AerialPhoto/gopro.MP4");


         if(!cap)
             exit;

      timer1 = new QTimer(this);
      connect(timer1, SIGNAL(timeout()), this, SLOT(updatepicture()));
      timer1->start(10);
 //CvCapture *cap1 =cvCreateFileCapture("http://10.5.5.9:8080/live/amba.m3u8");


  /*
    CvCapture *cap =cvCaptureFromFile("/Users/pranavagarwal/Desktop/AerialPhoto/gopro.MP4");


     if(!cap)
         exit;

     int frmcnt =(int)cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_COUNT);
     int fps =(int)cvGetCaptureProperty(cap,CV_CAP_PROP_FPS);

     //cvNamedWindow("Video");
     int i=0;


      printf("Entering Video --%d--%d----",frmcnt,fps);
     while (i<frmcnt)
     {

         IplImage* img =cvQueryFrame(cap);

        QImage qimgvid = QImage((const unsigned char*)img->imageData,img->width,img->height,QImage::Format_RGB888).rgbSwapped();
         qimgvid.scaled(550,450,Qt::IgnoreAspectRatio);
         ui->label->setPixmap(QPixmap::fromImage(qimgvid));
         ui->label->showFullScreen();

         /*IplImage *red1 = show_histogram(img,"red");
         IplImage *green1 =show_histogram(img,"green");
         IplImage *blue1 =show_histogram(img,"blue");
         red =QImage((const unsigned char*)red1->imageData,red1->width,red1->height,QImage::Format_RGB888).rgbSwapped();
         red.scaled(180,70,Qt::IgnoreAspectRatio);
         ui->red->setPixmap(QPixmap::fromImage(red));
         green =QImage((const unsigned char*)green1->imageData,green1->width,green1->height,QImage::Format_RGB888).rgbSwapped();
         green.scaled(180,70,Qt::IgnoreAspectRatio);
         ui->green->setPixmap(QPixmap::fromImage(green));
         blue =QImage((const unsigned char*)blue1->imageData,blue1->width,blue1->height,QImage::Format_RGB888).rgbSwapped();
         blue.scaled(180,70,Qt::IgnoreAspectRatio);
         ui->blue->setPixmap(QPixmap::fromImage(blue));*/


       /* cvShowImage("Video",img);
         if(cvWaitKey(10)==27)
             break;
         cvWaitKey(100);
         //cvReleaseImage(&img);

        i=i+1;
     }

    cvReleaseCapture(&cap);
    cvDestroyWindow("Video");*/

}

void TestOpencv::on_pushButton_10_clicked()
{

     QMessageBox::information(this,tr("The Title"),tr("Please Select An Aerial Video") );
   cap1 =cvCaptureFromFile("/Users/pranavagarwal/Desktop/Image/gopro1.mov");

    if(!cap1)
    {
     QMessageBox::information(this,tr("The Title"),tr("Error in Video Loading") );
     exit;
    }
    else
    {
        QMessageBox::information(this,tr("The Title"),tr("Aerial Video has been Loaded") );

    }
 }

void TestOpencv::on_verticalScrollBar_actionTriggered(int action)
{

    totalfrmnum =cvGetCaptureProperty(cap1,CV_CAP_PROP_FRAME_COUNT);
    int frmscale = totalfrmnum/100;
    int frmqury = frmscale * action;
    IplImage *frm =retrvframe(cap1,frmqury);
    IplImage *newfrm= cvCreateImage(cvSize(650,480),8,3);
    cvResize(frm,newfrm,CV_INTER_LINEAR);
    QImage qimgvid = QImage((const unsigned char*)newfrm->imageData,newfrm->width,newfrm->height,QImage::Format_RGB888).rgbSwapped();
    qimgvid.scaled(220,180,Qt::IgnoreAspectRatio);
    ui->label_2->setPixmap(QPixmap::fromImage(qimgvid));
    iplImg =newfrm;


}
