/*************************************************************************
	> File Name: knn.h
	> Author: ruihong
	> Mail: 
	> Created Time: 2017年05月13日 星期六 11时24分23秒
 ************************************************************************/

#ifndef _KNN_H
#define _KNN_H
#include<string>
#include<vector>
using namespace std;

//rating matrix
struct R_mat{
    int n_user;
    int n_item;
    vector<vector<int> > matrix;
    R_mat():n_user(0),n_item(0){};
};

struct ItemSimilarities{
    int n_item;
    vector<vector<float> > similar;
    ItemSimilarities():n_item(0){};
};

class ItemBasedKnn{
public:
    R_mat ratingMat;
    ItemSimilarities similarMat;
    vector<vector<float> > pred_res;//评分预测结果

    void getRatingMatrix(string path);
    ItemBasedKnn(int user,int item);
    //~ItemBasedKnn();

    //计算物品相似度
    void calItemSimilarity();

    //对未知评分进行预测,k为近邻数
    void pred(int n_threads,int k);
 

private:
    //用户与物品的最大编号
    int n_user;
    int n_item;
    vector<float> avg_ri;
    //vector<float> avg_ru;
    
    //计算物品i,j的距离
    void calPearsonDist(int i,int j);
};

#endif
