/*************************************************************************
	> File Name: main.cpp
	> Author: ruihong
	> Mail: 
	> Created Time: 2016年12月23日 星期五 21时48分34秒
 ************************************************************************/

#include<iostream>
#include<algorithm>
#include<fstream>
#include"dealdata.h"
using namespace std;
int main(){
    string datapath("../data/ml-1m/ratings.dat");
    int userhashsize = 6041;
    int itemhashsize = 3953;
    PrepareData getdata(datapath,userhashsize,itemhashsize);
    //IBasedData *item_based = getdata.getItemBasedData();
    UBasedData *user_based = getdata.getUserBasedData();
    //ofstream outtrain("../data/ml-1m/itembased.train.csv")
    //UBasedData *user_based = getdata.getUserBasedData();
    //ofstream output("../data/ml-1m/userbased.csv",ofstream::out);
    ofstream outtrain("../data/ml-1m/userbased.train.csv");
    ofstream outtrainMF("../data/ml-1m/userbased.trainMF.csv");
    ofstream outtest("../data/ml-1m/userbased.test.csv");
    int size = user_based->uhashtable.size();
    for(int i = 0; i < size; ++i){
        if(user_based->uhashtable[i] != NULL){
            outtrain<<i<<",";
            outtrainMF<<i<<",";
            outtest<<i<<",";
            sort(user_based->uhashtable[i]->begin(),user_based->uhashtable[i]->end());
            int sizej = (user_based->uhashtable[i])->size();
            int j = 0;
            for(; j < sizej-10; ++j){
                outtrain<<(*(user_based->uhashtable[i]))[j].movid<<",";
                outtrainMF<<(*(user_based->uhashtable[i]))[j].movid<<","<<(*(user_based->uhashtable[i]))[j].rating<<",";
                //output<<(*(user_based->uhashtable[i]))[j].time<<",";
            }
            outtrain<<(*(user_based->uhashtable[i]))[j].movid<<endl;
            outtrainMF<<(*(user_based->uhashtable[i]))[j].movid<<","<<(*(user_based->uhashtable[i]))[j].rating<<endl;
            j++;
            //output<<(*(user_based->uhashtable[i]))[j].time;
            for(; j < sizej; ++j){
                outtest<<(*(user_based->uhashtable[i]))[j].movid<<",";
            }
            outtest<<(*(user_based->uhashtable[i]))[j].movid<<endl;
        }
    }
    outtrain.close();
    outtrainMF.close();
    outtest.close();
    delete user_based;
    return 0;
}

