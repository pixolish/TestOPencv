
#include "testopencv.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    TestOpencv w;
    w.show();

    return a.exec();
}
