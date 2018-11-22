/*
*
* g++ -O3 ./code.cc `pkg-config --libs opencv`
* 
* 
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <unistd.h>
#include "rnd.h".  // choose your favorite RNG


using namespace std;
using namespace cv;

#define N_pi 3.1415

// Parameters
const int n_pixel = 100;
double T = 0;
double sig1 = 2;
double sig2 = 4;
double amp = 0*1;
const int n_theta = 64;


//Plotting and discretization
double colormap16[16][3] = {{0, 1, 0}, {0.1569, 1, 0},
{0.5490, 1, 0}, {0.8235, 1, 0}, {1, 1, 0}, {1, 0.8235, 0},
{1, 0.5882, 0},{1, 0.3529, 0}, {1, 0, 0}, {1, 0, 0.4784},
{0.7059, 0, 1}, {0.3922, 0, 1}, {0, 0, 1}, {0, 0.3137, 1},
{0, 0.5490, 0.5882}, {0, 0.7843, 0.1961}};

double colormap[64][3] = {{1, 0, 0}, {1, 0.0938, 0}, {1, 0.1875, 0},
{1, 0.2812, 0}, {1, 0.3750, 0}, {1, 0.4688, 0}, {1, 0.5625, 0},
{1, 0.6562, 0}, {1, 0.7500, 0}, {1, 0.8438, 0}, {1, 0.9375, 0},
{0.9688, 1, 0}, {0.8750, 1, 0}, {0.7812, 1, 0}, {0.6875, 1, 0},
{0.5938, 1, 0}, {0.5000, 1, 0}, {0.4062, 1, 0}, {0.3125, 1, 0},
{0.2188, 1, 0}, {0.1250, 1, 0}, {0.0312, 1, 0}, {0, 1, 0.0625},
{0, 1, 0.1562}, {0, 1, 0.2500}, {0, 1, 0.3438}, {0, 1, 0.4375},
{0, 1, 0.5312}, {0, 1, 0.6250}, {0, 1, 0.7188}, {0, 1, 0.8125},
{0, 1, 0.9062}, {0, 1, 1}, {0, 0.9062, 1}, {0, 0.8125, 1},
{0, 0.7188, 1}, {0, 0.6250, 1}, {0, 0.5312, 1}, {0, 0.4375, 1},
{0, 0.3438, 1}, {0, 0.2500, 1}, {0, 0.1562, 1}, {0, 0.0625, 1},
{0.0312, 0, 1}, {0.1250, 0, 1}, {0.2188, 0, 1}, {0.3125, 0, 1},
{0.4062, 0, 1}, {0.5000, 0, 1}, {0.5938, 0, 1}, {0.6875, 0, 1},
{0.7812, 0, 1}, {0.8750, 0, 1}, {0.9688, 0, 1}, {1, 0, 0.9375},
{1, 0, 0.8438}, {1, 0, 0.7500}, {1, 0, 0.6562}, {1, 0, 0.5625},
{1, 0, 0.4688}, {1, 0, 0.3750}, {1, 0, 0.2812}, {1, 0, 0.1875}, 
{1, 0, 0.0938}};


int clip(int s){
    if(s<1000){
        return s;
    }else{
        return 1000;
    }
}

void field_for_spin(int spin_index, int spins[n_pixel*n_pixel], double M[n_theta][n_theta], double exp_array[1001]){
    /* Calculates the Metropolis update for spin "spin_index" */
    int column_spin = spin_index%n_pixel;
    int row_spin =  (spin_index-column_spin)/n_pixel;
    int theta_new = rndint(n_theta);
    int theta_old = spins[spin_index];
    double e_new = 0;
    double e_old = 0;
    int lo = -2*sig2;
    int up =  2*sig2;
    for(int r=lo; r<up; r++){               
        for (int c=lo; c<up; c++){
            double dist2  = r*r + c*c;
            if (dist2>0){
                int cc  = (column_spin + c + n_pixel)%n_pixel;
                int rr = (row_spin + r + n_pixel)%n_pixel;
                int theta_comp = spins[cc + rr*n_pixel];
                int d1 =  clip( 250*dist2/(2*sig1*sig1) );
                int d2 =  clip( 250*dist2/(2*sig2*sig2) );
                double Jij =  (exp_array[d1]/(sig1*sig1)) - (amp*exp_array[d2]/(sig2*sig2));
                //double Jij =  (exp(-dist2/(2*sig1*sig1))/(sig1*sig1)) - (amp*exp(-dist2/(2*sig2*sig2))/(sig2*sig2));
                e_new -= Jij*M[theta_comp][theta_new];
                e_old -= Jij*M[theta_comp][theta_old];
            }
        }
    }
    if(e_new<e_old){
        spins[spin_index] = theta_new;
    }
    else if(T>0){
        double p = exp((e_old-e_new)/T);
        if(rnd()<p)
            spins[spin_index] = theta_new;
    }
}


int main(){

    //Initialization
    InitRandom();
    int spins[n_pixel*n_pixel];
    for(int i=0;i<n_pixel*n_pixel;i++){
        spins[i]=rndint(n_theta);
    }
    double M[n_theta][n_theta];
    for(int i=0; i<n_theta; i++){
        for(int j=0; j<n_theta; j++){
            M[i][j] = cos(2*N_pi*(1.*i/n_theta) - 2*N_pi*(1.*j/n_theta));
        }
    }
    double exp_array[1001];
    for(int r=0; r<1001; r++){
        double d = r/250.;
        exp_array[r] = exp(-d);   //Samples Gaussian from 0 to 4 in steps of 0.004
    }


    // Updates
    for (int t=0; t<20000*n_pixel*n_pixel; t=t+1){    // loop in which all spins are touched once.
        int spin_index = rndint(n_pixel*n_pixel);
        field_for_spin(spin_index, spins, M, exp_array);

        if(t%(n_pixel*n_pixel*2)==0){
            cout<<"Time [gens.] elapsed: "<<t/(n_pixel*n_pixel)<<endl;
            // Plot whenever, on average, every spin has been touched. 
            uint8_t x[n_pixel][n_pixel][3];
            for (int column=0; column<n_pixel; column++){
                for(int row=0; row<n_pixel; row++){
                    double th = spins[column*n_pixel + row];
                    int ind = 64*(th/n_theta);
                    x[row][column][0] = 255*colormap[ind][1]; //b
                    x[row][column][1] = 255*colormap[ind][2]; //g
                    x[row][column][2] = 255*colormap[ind][0]; //r
                }
            }
            const int scale = 1;
            cv::Mat spin_image(n_pixel, n_pixel, CV_8UC3, x), image_scaled;
            cv::resize(spin_image, image_scaled, cv::Size(spin_image.cols*5, spin_image.rows*5), 0,0, cv::INTER_NEAREST);
            cv::imshow("Spins", image_scaled);
            cv::waitKey(1);
        }
    }



    return 0;
}





