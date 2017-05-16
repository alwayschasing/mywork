/*************************************************************************
	> File Name: run.cpp
	> Author: ruihong
	> Mail: 
	> Created Time: 2017年05月14日 星期日 16时53分03秒
 ************************************************************************/

#include"knn.h"
#include"utility.h"
#include<iostream>
#include<fstream>
#include<vector>
using namespace std;


void getTestData(vector<vector<int> >& te_data){
    ifstream input("/home/lrh/graduation_project/data/ml-1m/userbased.test.csv");
    string line;
    while(!input.eof()){
        getline(input,line); 
        vector<int> temp = split2int(line,',');
        for(int i = 1; i < temp.size(); ++i){
            te_data[0][i-1] = temp[i];
        } 
    }
}

void recommend(vector<vector<int> >& rec_res,vector<vector<float> >& rating_pred){
    int ln = rating_pred.size();
    for(int i = 1; i < ln; ++i){
        vector<int> rec = argsort(rating_pred[i]);
        rec_res[i] = rec;
    }
}

float eval(vector<vector<int> >& te_data,vector<vector<int> >& rec){
    float recall = 0.0;
    int hit = 0;
    int ln = te_data.size(); 
    int n_rec = te_data[0].size();
    int hituser = 0;
    for(int i = 1; i < ln; ++i){
    //统计每个用户的召回率,用户编号从1开始
        hit = 0;
        for(int p = 0; p < n_rec; ++p){
            for(int q = 0; q < n_rec; ++q){
                if(rec[i][p] == te_data[i][q]) hit++;
            }
        }
        if(hit == 0) continue;
        hituser++;
        recall += ((float)hit/n_rec); 
    }
    recall = recall/(ln-1);
    return recall;
}

int main(){
    string tr_dataPath = "/home/lrh/graduation_project/data/ml-1m/userbased.trainMF.csv";
    int n_thread = 40;
    int n_user = 6040;
    int n_item = 3952;
    ItemBasedKnn itemknn(n_user,n_item);
    itemknn.getRatingMatrix(tr_dataPath);
    itemknn.calItemSimilarity();
    itemknn.pred(n_thread,5);

    vector<vector<int> > rec_res(n_user+1,vector<int>(10,0));
    recommend(rec_res,itemknn.pred_res); 

    vector<vector<int> > te_data(n_user+1,vector<int>(10,0)); 
    getTestData(te_data);

    float recall = 0.0;
    recall = eval(te_data,rec_res);
    return 0; 
}

