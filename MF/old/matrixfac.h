/*************************************************************************
	> File Name: matrixfac.h
	> Author: ruihong
	> Mail: 
	> Created Time: 2016年12月23日 星期五 18时41分05秒
 ************************************************************************/

#ifndef _MATRIXFAC_H
#define _MATRIXFAC_H
#include<vector>
#include"../preparedata/dealdata.h"
using namespace std;
class MF {
public:
    vector<vector<float>*> getItemFactor();
private:
    vector<vector<float>*> itemfactor;
    IBasedData *data;
};
#endif
