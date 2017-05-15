/*************************************************************************
	> File Name: itemBased.cpp
	> Author: ruihong
	> Mail: 
	> Created Time: 2017年05月13日 星期六 20时49分39秒
 ************************************************************************/

#include"knn.h"
#include"utility.h"
#include<iostream>
#include<fstream>
#include<memory.h>
#include<vector>
#include<pthread.h>
#include<math.h>
using namespace std;

//读取训练文件构造rating矩阵并返回
R_mat ItemBasedKnn::getRatingMatrix(string path){
    ifstream input(path,ifstream::in);
    string line;
    while(!input.eof()){
        getline(input,line); 
        vector<int> temp = split2int(line,','); 
        int user = temp[0];
        int n = temp.size()/2;
        for(int i = 0; i < n; ++i){
            int item = temp[(i<<1)+1];
            int rating = temp[(i<<2)+2];
            ratingMat.matrix[user][item] = rating;
        }
    }
    return ratingMat;
}
//构造函数
ItemBasedKnn::ItemBasedKnn(int user,int item):n_user(user),n_item(item){
    //评分矩阵初始化
    ratingMat.n_user = n_user;
    ratingMat.n_item = n_item;
    ratingMat.matrix = vector<vector<int>>(n_user+1,vector<int>(n_item+1,0));

    //物品相似矩阵初始化
    similarMat.n_item = n_item;
    similarMat.similar = vector<vector<float>>(n_item+1,vector<float>(n_item+1,0.0));
    
    //平均值初始化
    avg_ri = vector<float>(n_item+1,0.0);
    //avg_ru = vector<float>(n_user+1,0.0);

    //预测结果初始化
    pred_res = vector<vector<float>>(n_user+1,vector<float>(n_item+1,0.0));
}
//析构函数
/*
ItemBasedKnn::~ItemBasedKnn(){
}
*/

//计算物品相似度,得到物品相似度矩阵
void ItemBasedKnn::calPearsonDist(int i,int j){
    vector<int> u_set;
    for(int u = 1; i <= n_user; ++u){
        if(ratingMat.matrix[u][i]&&ratingMat.matrix[u][j]){
            u_set.push_back(u);
        }
    }
    float a = 0.0; //分子
    float b1 = 0.0; //分母
    float b2 = 0.0;
    for(int u:u_set){
        a += (ratingMat.matrix[u][i]-avg_ri[i])*(ratingMat.matrix[u][j]-avg_ri[j]);
        b1 += (ratingMat.matrix[u][i]-avg_ri[i])*(ratingMat.matrix[u][i]-avg_ri[i]);
        b2 += (ratingMat.matrix[u][j]-avg_ri[j])*(ratingMat.matrix[u][j]-avg_ri[j]);
    }
    
    similarMat.similar[i][j] = a/(sqrt(b1)*sqrt(b2)); 
}
void ItemBasedKnn::calItemSimilarity(){
    //计算物品评分均值
    for(int i = 1; i <= n_item; ++i){
       int sum = 0;
       for(int j = 1; j <= n_user; ++j){
           sum += ratingMat.matrix[j][i];
       }
       avg_ri[i] = float(sum)/n_user;
    }
    for(int i = 1; i <= n_item; ++i){
        for(int j = i+1; j <= n_item; ++j){
            calPearsonDist(i,j);
        }
    }
    for(int i = n_item; i >= 1; ++i){
        for(int j = i-1; j >= 1; --j){
            similarMat.similar[i][j] = similarMat.similar[j][i];
        }
    }
}

/*
使用多线程的方式对预测过程进行处理
*/
//线程工作函数的参数设置
struct parameter{
    int start; //批数据中首个用户
    int batch_size;
    vector<vector<int>> *mat;
    vector<vector<float>> *sim;
    vector<vector<float>> *res;
    int k;
};
//线程工作函数
void *threadWork(void *arg){
    parameter *p = (parameter*)arg;
    int start = p->start;
    int batch_size = p->batch_size;
    vector<vector<int>> *mat = p->mat;
    vector<vector<float>> *sim = p->sim;
    vector<vector<float>> *res = p->res;
    int k = p->k;
    int end = start+batch_size-1;
    int n_item = (*sim)[0].size()-1;

    for(int i = start; i <= end; ++i){
        //处理每个用户
        for(int j = 1; j <=n_item; ++j){
            if((*res)[i][j] == 0){
                int neighbor = 0;
                vector<int> indexs = argsort((*sim)[i]); 
                vector<int> neighbor_set(k,0);
                for(int index:indexs){
                    if((*mat)[i][index] != 0){
                        neighbor_set[neighbor] = index; 
                        neighbor++;
                    }
                    if(neighbor>=k) break;
                }
                if(neighbor < k) continue;
                float a = 0.0;
                for(int p = 0; p < k; p++){
                    a = a + (*sim)[i][neighbor_set[p]]*(*mat)[i][neighbor_set[p]];
                }
                float b = 0.0;
                for(int p = 0; p < k; p++){
                    b += fabs((*sim)[i][neighbor_set[p]]);
                }
                (*res)[i][j] = a/b;
            }
        }
    }
    
    pthread_exit(0);
}
//对未知评分进行预测,使用多线程,参数为线程数
void ItemBasedKnn::pred(int n_threads,int k){
    int batch_size = n_user/n_threads;//每个线程处理的用户数 
    pthread_t *pt = (pthread_t *)malloc(n_threads*sizeof(pthread_t));

    int start = 0; //记录每个线程所处理数据的开始用户编号,从1开始
    //留出一个线程来处理最后一批用户
    for(int i = 0; i < n_threads-1; ++i){
        start = 1+batch_size*i;
        parameter arg = {start,batch_size,&ratingMat.matrix,&similarMat.similar,&pred_res,k};
        pthread_create(&pt[i],NULL,threadWork,&arg);
    }
    //对最后一批用户特殊处理
    parameter arg = {start,n_user-start,&ratingMat.matrix,&similarMat.similar,&pred_res,k};
    pthread_create(&pt[n_threads-1],NULL,threadWork,&arg);

    for(int i = 0; i < n_threads; ++i) pthread_join(pt[i],NULL);
}

