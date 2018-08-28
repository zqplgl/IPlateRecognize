//
// Created by zqp on 18-8-28.
//
#include <IPlateRecognize.h>

using namespace Vehicle;

void getImPath(string& pic_dir,vector<string>&impaths)
{
    string cmd = "find "+pic_dir+" -name *.jpg";
    FILE *fp = popen(cmd.c_str(),"r");
    char buffer[512];
    while(1)
    {
        fgets(buffer,sizeof(buffer),fp);
        if(feof(fp))
            break;
        buffer[strlen(buffer)-1] = 0;
        impaths.push_back(string(buffer));
    }
}

int main()
{
    string model_dir = "/home/zqp/install_lib/models";
    IPlateRecognize *detector = CreateIPlateRecognize(model_dir,0);

    string pic_dir = "/home/zqp/testimage";
    vector<string> impaths;
    getImPath(pic_dir,impaths);

    vector<PlateInfo> plateinfos;
    for(int i=0; i<impaths.size(); ++i)
    {
        cv::Mat im = cv::imread(impaths[i]);
        detector->detect(im,plateinfos,0.7);

        for(int j =0; j<plateinfos.size(); ++j)
        {
            cout<<"license: "<<plateinfos[j].license<<"\t"<<
                "color: "<<plateinfos[j].color<<"\t"<<
                "score: "<<plateinfos[j].score<<"\t"<<endl;
            cv::rectangle(im,plateinfos[j].zone,cv::Scalar(0,0,255));
        }

//        cv::imshow("im",im);
//        cv::waitKey(0);
    }




}

