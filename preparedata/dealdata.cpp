/*************************************************************************
	> File Name: dealdata.cpp
	> Author: ruihong
	> Mail: 
	> Created Time: 2016年12月20日 星期二 21时05分32秒
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<regex>
#include"dealdata.h"

UBasedData::UBasedData(int hs):uhashtable(vector<vector<uunit>*>(hs,NULL)),hashsize(hs){
    //usernumber = unum;
    //hashsize = hs;
    //uhashtable = vector<vector<uunit>*>(hs,NULL);
}
void UBasedData::setUserNum(int unum){
    usernumber = unum;
}
int UBasedData::getUserNum(){
    return usernumber;
}
PrepareData::PrepareData(string rawdatapath,int userhashsize,int itemhashsize)
    :datapath(rawdatapath),user_hash_size(userhashsize),item_hash_size(itemhashsize){
    //datapath = rawdatapath;
    //user_hash_size = userhashsize;
    //item_hash_size = itemhashsize;
}
UBasedData::~UBasedData(){
    int size = uhashtable.size();
    for(int i = 0; i < size; ++i){
        if(uhashtable[i] != NULL){
            delete uhashtable[i];
        }
    }
}
IBasedData::IBasedData(int hs):ihashtable(vector<vector<iunit>*>(hs,NULL)),hashsize(hs){}
IBasedData::~IBasedData(){
    int size = ihashtable.size();
    for(int i = 0; i < size; ++i){
        if(ihashtable[i] != NULL){
            delete ihashtable[i];
        }
    }
}
//the uunit and iunit initial function
uunit::uunit(int movid,int rating,int time):movid(movid),rating(rating),time(time){}
iunit::iunit(int userid,int rating,int time):userid(userid),rating(rating),time(time){}

UBasedData* PrepareData::getUserBasedData(){
    regex reg(".*\\.dat$");
    //detect the delimiter character befor user and item
    if(regex_match(datapath,reg)) delim = ':';
    ifstream input(datapath);
    char *title = new char[35];
    char *dm = new char[3];
    int userid;
    int itemid;
    int rating;
    int timestamp;
    //define the UBasedData 
    UBasedData* u2item = new UBasedData(user_hash_size);
    //process the .dat file
    if(delim == ':'){
        while(!input.eof()){
            input>>userid;
            input.get(dm,3);
            input>>itemid;
            input.get(dm,3);
            input>>rating;
            input.get(dm,3);
            input>>timestamp;
            input.get();
            //cout<<userid<<','<<itemid<<','<<rating<<','<<timestamp<<endl;
            if(u2item->uhashtable[userid] == NULL){
                u2item->uhashtable[userid] = new vector<uunit>();
                (*u2item->uhashtable[userid]).push_back(uunit(itemid,rating,timestamp));
            }else{ 
                (*u2item->uhashtable[userid]).push_back(uunit(itemid,rating,timestamp));
            }
        }
    } 
    //precess the .csv file
    else if(delim == ','){
        input.getline(title,35);
        while(!input.eof()){    
            input>>userid;
            input.get();
            input>>itemid;
            input.get();
            input>>rating;
            input.get();
            input>>timestamp;
            input.get();
            //cout<<userid<<','<<itemid<<','<<rating<<','<<timestamp<<endl;
            if(u2item->uhashtable[userid] == NULL){
                u2item->uhashtable[userid] = new vector<uunit>();
                (*u2item->uhashtable[userid]).push_back(uunit(itemid,rating,timestamp));
            }else{ 
                (*u2item->uhashtable[userid]).push_back(uunit(itemid,rating,timestamp));
            }
        }
    }
    else cout<<"input get errors"<<endl;
    input.close();
    delete [] title;
    delete [] dm;
    return u2item;
}

IBasedData* PrepareData::getItemBasedData(){
    
    regex reg(".*\\.dat$");
    //detect the delimiter character befor user and item
    if(regex_match(datapath,reg)) delim = ':';
    ifstream input(datapath);
    char *title = new char[35];
    char *dm = new char[3];
    int userid;
    int itemid;
    int rating;
    int timestamp;

    IBasedData* i2user = new IBasedData(item_hash_size);
    if(delim == ':'){
        while(!input.eof()){
            input>>userid;
            input.get(dm,3);
            input>>itemid;
            input.get(dm,3);
            input>>rating;
            input.get(dm,3);
            input>>timestamp;
            input.get();
            //cout<<userid<<','<<itemid<<','<<rating<<','<<timestamp<<endl;
            if(i2user->ihashtable[itemid] == NULL){
                i2user->ihashtable[itemid] = new vector<iunit>();
                (*i2user->ihashtable[itemid]).push_back(iunit(userid,rating,timestamp));
            }else{ 
                (*i2user->ihashtable[itemid]).push_back(iunit(userid,rating,timestamp));
            }
        }
    
    }
    else if(delim == ','){
        input.getline(title,35);
        while(!input.eof()){    
            input>>userid;
            input.get();
            input>>itemid;
            input.get();
            input>>rating;
            input.get();
            input>>timestamp;
            input.get();
            //cout<<userid<<','<<itemid<<','<<rating<<','<<timestamp<<endl;
            if(i2user->ihashtable[userid] == NULL){
                i2user->ihashtable[userid] = new vector<iunit>();
                (*i2user->ihashtable[itemid]).push_back(iunit(userid,rating,timestamp));
            }else{ 
                (*i2user->ihashtable[itemid]).push_back(iunit(userid,rating,timestamp));
            }
        } 
    }
    else cout<<"input get error"<<endl;
    input.close();
    delete [] title;
    delete [] dm;
    return i2user;
}
