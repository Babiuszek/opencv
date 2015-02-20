/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

//C:\Users\Mateusz\Desktop\Computer Vision\OpenCV 3.0.0alpha\sources\modules\ml\src
#include "precomp.hpp"
#include "gcgraph.hpp"
#include <limits>

#include <iostream>

using namespace cv;

/*
This is implementation of image segmentation algorithm GrabCut described in
"GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts".
Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec3d color ) const;
    double operator()( int ci, const Vec3d color ) const;
    int whichComponent( const Vec3d color ) const;

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3];
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};
GMM::GMM( Mat& _model )
{
    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;

    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 3*componentsCount;

    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
             calcInverseCovAndDeterm( ci );
}
double GMM::operator()( const Vec3d color ) const
{
    double res = 0;
	// Sum of all components coefs * (
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}
double GMM::operator()( int ci, const Vec3d color ) const
{
    double res = 0;
	// If possibility exists
    if( coefs[ci] > 0 )
    {
		// Make sure Determ is greater then epsilon
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
		// Find the difference of our color to the mean of that particular component
        Vec3d diff = color;
        double* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
		// Calculate multiplier (component sum(inverseCov*diff)
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        // Calculate return value?
		res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}
int GMM::whichComponent( const Vec3d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}
void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}
void GMM::addSample( int ci, const Vec3d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}
void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            coefs[ci] = (double)n/totalSampleCount;

            double* m = mean + 3*ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

            double* c = cov + 9*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

            double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            if( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}
void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
        double *c = cov + 9*ci;
        double dtrm =
              covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}

/*
 GMM - Gaussian Mixture Model 6 dimensional
*/
class GMM6d
{
public:
    static const int componentsCount = 5;
	static const int dimension = 6;

    GMM6d( Mat& _model );
    double operator()( const Vec6d color ) const;
    double operator()( int ci, const Vec6d color ) const;
    int whichComponent( const Vec6d color ) const;

    void initLearning();
    void addSample( int ci, const Vec6d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][dimension][dimension];
    double covDeterms[componentsCount];

    double sums[componentsCount][dimension];
    double prods[componentsCount][dimension][dimension];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};
GMM6d::GMM6d( Mat& _model )
{
    const int modelSize = dimension/*mean*/ + dimension*dimension/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 43*componentsCount" );

    model = _model;

	// coefs has size of ComponentCount (each component has its own coefficient)
    coefs = model.ptr<double>(0);
	// Mean has the size of Dimension*ComponentCount (each dimension has its own mean for each component)
    mean = coefs + componentsCount;
	// The rest is simply taken by Covariance matrices
    cov = mean + dimension*componentsCount;

	// Initialize our GMMs
    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
             calcInverseCovAndDeterm( ci );
}
double GMM6d::operator()( const Vec6d color ) const
{
    double res = 0;
	// Sum of all components coefs * (
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}
double GMM6d::operator()( int ci, const Vec6d color ) const
{
    double res = 0;
	// If possibility exists
    if( coefs[ci] > 0 )
    {
		// Make sure Determ is not 0, the matrix is not singular
        CV_Assert( covDeterms[ci] != 0 );
		// Find the difference of our color to the mean of that particular component
        Vec6d diff = color;
        double* m = mean + dimension*ci;
		for (int i = 0; i < dimension; i++)
			diff[i] -= m[i];
		// Calculate multiplier (component sum(inverseCov*diff)
		/*
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
				   */
		// The below loop simulates the formula above expanded into n dimensions
		double mult = 0.0;
		for (int i = 0; i < dimension; i++)
			for (int j = 0; j < dimension; j++)
				mult += diff[i]*diff[j]*inverseCovs[ci][j][i];
        // Calculate return value?
		res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}
int GMM6d::whichComponent( const Vec6d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}
void GMM6d::initLearning()
{
	// Initialize every value we care about to 0
    for( int ci = 0; ci < componentsCount; ci++)
    {
		// All sums and prods of that component are set to 0
		for (int i = 0; i < dimension; i++)
		{
			sums[ci][i] = 0;
			for (int j = 0; j < dimension; j++)
				prods[ci][i][j] = 0;
		}
		// Same for it's sample counts
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}
void GMM6d::addSample( int ci, const Vec6d color )
{
	// Add to sums
	for (int i = 0; i < dimension; i++)
		sums[ci][i] += color[i];
	// Change prods
	for (int i = 0; i < dimension; i++)
		for (int j = 0; j < dimension; j++)
			prods[ci][i][j] += color[i]*color[j];
	// Increase our sample counters
    sampleCounts[ci]++;
    totalSampleCount++;
}
void GMM6d::endLearning()
{
	// Learning is done
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
		// Set the sample count. If a cluster is empty, just set it's coef
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
			// If it's not empty, we need to calculate it's means and covariance matrix
            coefs[ci] = (double)n/totalSampleCount;

			// Means, initialize the pointer
            double* m = mean + dimension*ci;
			for (int i = 0; i < dimension; i++)
				m[i] = sums[ci][i]/n; // Set all the means

			// Covariance matrix
            double* c = cov + dimension*dimension*ci;
			// The loops below simulate the formulas above for more dimensions
			// cov(X, Y) = E(X * Y) - EX * EY; where E is expected value
			for (int i = 0; i < dimension; i++)
				for (int j = 0; j < dimension; j++)
					c[dimension*i+j] = prods[ci][i][j]/n - m[i]*m[j];

			// Calculate determinant
			// We first write our data into a temporary Mat
			Mat c_mat;
			c_mat.create(dimension, dimension, CV_64FC1);
			for (int i = 0; i < dimension; i++)
				for (int j = 0; j < dimension; j++)
					c_mat.at<double>(i, j) = c[dimension*i+j];
			
			//double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
			double dtrm = determinant(c_mat);
			// And check for singular
            //if( dtrm <= std::numeric_limits<double>::epsilon() )
			if ( dtrm == 0 )
            {
                // Adds the white noise to avoid singular covariance matrix.
				// We change the diagonal
				for (int i = 0; i < dimension*dimension; i = i + dimension + 1)
					c[i] += variance;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}
void GMM6d::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
		// Initialize our covariance matrix
        double *c = cov + dimension*dimension*ci;
		
		// Calculate determinant
		// We first write our data into a temporary Mat
		Mat c_mat;
		c_mat.create(dimension, dimension, CV_64FC1);
		for (int i = 0; i < dimension; i++)
			for (int j = 0; j < dimension; j++)
				c_mat.at<double>(i, j) = c[dimension*i+j];
        //double dtrm =
        //      covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
		covDeterms[ci] = determinant(c_mat);
		CV_Assert( covDeterms[ci] > 0 );

		// Calculate inverse covariance matrix
		Mat c_inv;
		c_inv.create(dimension, dimension, CV_64FC1);
		double ans = invert(c_mat, c_inv);
		// Make sure not singular, using invert function output;
        CV_Assert( ans != 0 );

		// Copy ou answer to GMM data
		for (int i = 0; i < dimension; i++)
			for (int j = 0; j < dimension; j++)
				inverseCovs[ci][i][j] = c_inv.at<double>(i, j);
    }
}

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x>0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
	// gamma is 50 and used for straight edges, gammaDivSqrt2 is used for diagonal ones
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);

	// Create materials, each having amount of vertices equal to our img
	// CV_64FC1 <- this defines a vector of 1 value of 64-bit float (double), used for defining Materials
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );

	// Main loop
	// As these things are, they are equal to roughly:
	// W = ( gamma / dist(i,j) ) * exp ( - diff.dot(diff) / 2*avg(diff.dot(diff)) )
	// Where diff is the difference between color vectors
	// Note: We apply a gauss with variance equal to 1/2*avg difference in color
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3d color = img.at<Vec3b>(y,x);
            if( x-1>=0 ) // left
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

/*
  Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equel"
                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

/*
  Initialize mask using rectangular.
*/
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
	// Always 10 iterations, always clustering into 2 centers (logical)
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

	// Create vectors representing sumples and push our points into proper containers
    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
	// Standard debug, none should be empty
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );

	// Transform vector of Vec3f into an actual 2D material
	// Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP) <- probably this one
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
	// Run the K-means algorythm
	// (_data = bgdSamples, K=componentsCount(5), _bestLabels=bgdLabels(output),
	//	TermCriteria, attempts=0, flags=kMeansType, _centers=noArray(default))
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

	// Do the same for FGD...
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

	// Initialize GMMs learning
    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

/*
  Assign GMMs components for each pixel.
*/
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

/*
  Learn GMMs parameters.
*/
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
	// For each component count of our GMMs...
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
		// Check each point of our img...
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
				// And check if this particular component is equal to the best one
                if( compIdxs.at<int>(p) == ci )
                {
					// If it is, add this one as a sample to BGD / FGD GMM (choose the fitting one)
					// Add to the best component only
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

/*
  Construct GCGraph
*/
static void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                       GCGraph<double>& graph )
{
	// Initialize our graph values
    int vtxCount = img.cols*img.rows,
		// Edges go left, upleft, up, right. That is why we substract cols+rows 3 times
		// Left		-> Extra rows
		// Upleft	-> Extra rows + cols
		// Up		-> Extra cols
		// Upright	-> Extra rows + cols
		// We then substract corners twice, need to add them back hence the + 2
		// Each edge is double way (to and from) for Ford Fulkerson, so we multiply the total by 2
        edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
	// Create our graph
    graph.create(vtxCount, edgeCount);

	// Main loop, iterates for each point
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
			// Note - which is sink and which is source has no meaning, just the segmentation
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
				// It is inconclusive, set it's edges as derived from GMMs (using log likelyhood for that color)
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights( vtxIdx, fromSource, toSink );

            // set n-weights
			// Derive the edge values from previously calculated Material matrices
            if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}

//=============================[ 6D versions of above functions ]==============================
static double calcBeta6d( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec6d color = img.at<Vec6d>(y,x);
            if( x>0 ) // left
            {
                Vec6d diff = color - img.at<Vec6d>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec6d diff = color - img.at<Vec6d>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec6d diff = color - img.at<Vec6d>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec6d diff = color - img.at<Vec6d>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

static void calcNWeights6d( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
	// gamma is 50 and used for straight edges, gammaDivSqrt2 is used for diagonal ones
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);

	// Create materials, each having amount of vertices equal to our img
	// CV_64FC1 <- this defines a vector of 1 value of 64-bit float (double), used for defining Materials
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );

	// Main loop
	// As these things are, they are equal to roughly:
	// W = ( gamma / dist(i,j) ) * exp ( - diff.dot(diff) / 2*avg(diff.dot(diff)) )
	// diff.dot(diff) / 2*avg(diff.dot(diff)) changes from 0 to infinity (theoretically)
	// Where diff is the difference between color vectors
	// Note: We apply a gauss with variance equal to 1/2*avg difference in color
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec6d color = img.at<Vec6d>(y,x);
            if( x-1>=0 ) // left
            {
                Vec6d diff = color - img.at<Vec6d>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec6d diff = color - img.at<Vec6d>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec6d diff = color - img.at<Vec6d>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec6d diff = color - img.at<Vec6d>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

// TO DO: Create 6-dimensional GMMs so thy can accept such samples
static void initGMMs6d( const Mat& img, const Mat& mask, GMM6d& bgdGMM, GMM6d& fgdGMM )
{
	// Always 10 iterations, always clustering into 2 centers (logical)
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

	// Create vectors representing sumples and push our points into proper containers
    Mat bgdLabels, fgdLabels;
    std::vector<Vec6f> bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec6f)img.at<Vec6d>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec6f)img.at<Vec6d>(p) );
        }
    }
	// Standard debug, none should be empty
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
	
	// Transform vector of Vec3f into an actual 2D material
	// Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP) <- probably this one
    Mat _bgdSamples( (int)bgdSamples.size(), 6, CV_32FC1, &bgdSamples[0][0] );
	// Run the K-means algorythm
	// (_data = bgdSamples, K=componentsCount(5), _bestLabels=bgdLabels(output),
	//	TermCriteria, attempts=0, flags=kMeansType(2), _centers=noArray(default))
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

	// Do the same for FGD...
	Mat _fgdSamples( (int)fgdSamples.size(), 6, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

	// Learn GMMs
    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

static void assignGMMsComponents6d( const Mat& img, const Mat& mask, const GMM6d& bgdGMM, const GMM6d& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec6d color = img.at<Vec6d>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

static void learnGMMs6d( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM6d& bgdGMM, GMM6d& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec6d>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec6d>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

static void constructGCGraph6d( const Mat& img, const Mat& mask, const GMM6d& bgdGMM, const GMM6d& fgdGMM, double lambda,
                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                       GCGraph<double>& graph )
{
	// Initialize our graph values
    int vtxCount = img.cols*img.rows,
		// Edges go left, upleft, up, right. That is why we substract cols+rows 3 times
		// Left		-> Extra rows
		// Upleft	-> Extra rows + cols
		// Up		-> Extra cols
		// Upright	-> Extra rows + cols
		// We then substract corners twice, need to add them back hence the + 2
		// Each edge is double way (to and from) for Ford Fulkerson, so we multiply the total by 2
        edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
	// Create our graph
    graph.create(vtxCount, edgeCount);

	// Main loop, iterates for each point
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec6d color = img.at<Vec6d>(p);

            // set t-weights
			// Note - which is sink and which is source has no meaning, just the segmentation
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
				// It is inconclusive, set it's edges as derived from GMMs (using log likelyhood for that color)
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights( vtxIdx, fromSource, toSink );

            // set n-weights
			// Derive the edge values from previously calculated Material matrices
            if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}
//==================================[ End of 6D functions ]====================================

/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    graph.maxFlow();
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

// Shrinking function, creates a material 10 times smaller
Mat* shrink( const Mat& input, const int by )
{
	// First let us make sure the image can be correctly shrunk as given
	if (div(input.rows,by).rem != 0 || div(input.cols,by).rem != 0)
        CV_Error( CV_StsBadArg, "Input is of wrong size." );

	// Create our shrunk Material
	Mat* output = new Mat(input.rows/by, input.cols/by, CV_64FC(6));

	// For each point in ouput image...
	Point p_i; // Input iterator
	Point p_o; // Output iterator
    for( p_o.y = 0; p_o.y < output->rows; p_o.y++ )
    {
        for( p_o.x = 0; p_o.x < output->cols; p_o.x++ )
		{
			// ...we take it's values (vector of 6 values, 3 colors and 3 standard deviations)
			Vec6d& values = output->at<Vec6d>(p_o);
			// And calculate these values using the area from input image. Wefirst calculate the means
			values[0] = 0.0;
			values[1] = 0.0;
			values[2] = 0.0;
			for ( p_i.y = by*p_o.y; p_i.y < by*(p_o.y+1); p_i.y++)
			{
				for ( p_i.x = by*p_o.x; p_i.x < by*(p_o.x+1); p_i.x++)
				{
					Vec3d color = (Vec3d)input.at<Vec3b>(p_i);
					values[0] += color.val[0];
					values[1] += color.val[1];
					values[2] += color.val[2];
				}
			}
			values[0] /= by*by;
			values[1] /= by*by;
			values[2] /= by*by;
			
			// Then calculate the standard deviations
			values[3] = 0.0;
			values[4] = 0.0;
			values[5] = 0.0;
			for ( p_i.y = by*p_o.y; p_i.y < by*(p_o.y+1); p_i.y++)
			{
				for ( p_i.x = by*p_o.x; p_i.x < by*(p_o.x+1); p_i.x++)
				{
					Vec3d color = (Vec3d)input.at<Vec3b>(p_i);
					values[3] += pow(values[0] - color.val[0], 2);
					values[4] += pow(values[1] - color.val[1], 2);
					values[5] += pow(values[2] - color.val[2], 2);
				}
			}
			values[3] = sqrt(values[3] / (by*by));
			values[4] = sqrt(values[4] / (by*by));
			values[5] = sqrt(values[5] / (by*by));
		}
	}

	// And done!
	return output;
}

// Function that creates a material of 3 rounded colors from shrunk means, as desired by GMMs
Mat* round_shrunk(const Mat& mask)
{
	Mat* output = new Mat(mask.rows, mask.cols, CV_8UC3);
	Point p;
    for( p.y = 0; p.y < output->rows; p.y++ )
    {
        for( p.x = 0; p.x < output->cols; p.x++ )
		{
			Vec6d vm = mask.at<Vec6d>(p);
			Vec3b& vo = output->at<Vec3b>(p);

			vo[0] = (uchar)vm[0];
			vo[1] = (uchar)vm[1];
			vo[2] = (uchar)vm[2];
		}
	}
	return output;
}

// Function that copies the answer from shrunk image onto the bigger mask
void expandShrunkMat(const Mat& input, Mat& output, const int by)
{
	Point p_i; // Input iterator
	Point p_o; // Output iterator
	
	// Go through each point of shrunk, input mask...
    for( p_i.y = 0; p_i.y < input.rows; p_i.y++ )
	{
        for( p_i.x = 0; p_i.x < input.cols; p_i.x++ )
		{
			uchar flag = input.at<uchar>(p_i);
			// And set its value onto every single pixel from a corresponding area on output
			for ( p_o.y = by*p_i.y; p_o.y < by*(p_i.y+1); p_o.y++)
				for ( p_o.x = by*p_i.x; p_o.x < by*(p_i.x+1); p_o.x++)
					output.at<uchar>(p_o) = flag;
		}
	}
	// End of input for
}

void grabCutInitShrunk( InputArray _img, InputOutputArray _mask, Rect _rect,
                  int iterCount, int mode )
{
	const int by = 10;

	// Standard null checking procedure
    if( _img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( _img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

	// Initialization
    Mat* img6d = shrink(_img.getMat(), by); // Shrunk image
	Mat* img = round_shrunk(*img6d);
    Mat mask; // Shrunk mask
	Mat& out_mask = _mask.getMatRef();
	Rect rect = Rect(_rect.x/by, _rect.y/by,
					min((int)ceil((double)_rect.width/(double)by), img->cols-1), //Max possible amount of rectangles
					min((int)ceil((double)_rect.height/(double)by), img->rows-1)); // Shrunk rect
    Mat bgdModel = Mat(); // Our local model
    Mat fgdModel = Mat(); // Same as above

	// Building GMMs for local models
    GMM6d bgdGMM( bgdModel ), fgdGMM( fgdModel );
    Mat compIdxs( img->size(), CV_32SC1 );

	// Here we always initialize with rect
    initMaskWithRect( mask, img->size(), rect );
    initGMMs6d( *img6d, mask, bgdGMM, fgdGMM );

	// Check mask for any errors
    if( mode == GC_EVAL )
        checkMask( *img, mask );

	// Simple parameters of our algorythm, used for setting up edge flows
    const double gamma = 50; // Gamma seems to be just a parameter for lambda, here 50
    const double lambda = 9*gamma; // Lambda is simply a max value for flow, be it from source or to target, here 450
    const double beta = calcBeta6d( *img6d ); // Beta is a parameter, here 1/(2*avg(sqr(||color[i] - color[j]||)))
										 // 1 / 2*average distance in colors between neighbours

	// NWeights, the flow capacity of our edges
    Mat leftW, upleftW, upW, uprightW;
    calcNWeights6d( *img6d, leftW, upleftW, upW, uprightW, beta, gamma );

	// The main loop
    for( int i = 0; i < iterCount; i++ )
    {
		// Simply initialize the graph we will be using throughout the algorythm. It is created empty
        GCGraph<double> graph;
		
		// Check the image at mask, and depending if it's FGD or BGD, return the number of component that suits it most.
		// Answer (component numbers) is stored in compIdxs, it does not store anything else
        assignGMMsComponents6d( *img6d, mask, bgdGMM, fgdGMM, compIdxs );

		// This one adds samples to proper GMMs based on best component
		// Strengthens our predictions?
        learnGMMs6d( *img6d, mask, compIdxs, bgdGMM, fgdGMM );

		// NOTE: As far as I can tell these two will be primarily worked upon
		// Construct grapg cut graph, including initializing s and t values for source and sink flows
        constructGCGraph6d( *img6d, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );

		// Using max flow algorythm calculate segmentation
        estimateSegmentation( graph, mask );
    }

	// Piece together full size mask out of smaller one
	expandShrunkMat(mask, out_mask, by);
	
	// Clean-up
	delete img;
	delete img6d;
}

// Main grabcut algorythm
void cv::grabCut( InputArray _img, InputOutputArray _mask, Rect rect,
                  InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                  int iterCount, int mode )
{
	if (mode == GC_INIT_SHRUNK)
	{
		grabCutInitShrunk(_img, _mask, rect, iterCount, mode);
		return;
	}

	// Initialization
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef(); // Our answer, array of BGD/PR_BGD/PR_FGD/FGD
    Mat& bgdModel = _bgdModel.getMatRef(); // It is not changed outside, only in grabcut
    Mat& fgdModel = _fgdModel.getMatRef(); // Same as above

	// Standard null checking procedure
    if( img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );

	// Building GMMs, they are discarded after each cycle
    GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );
    Mat compIdxs( img.size(), CV_32SC1 );

	// Mask is either initialized via rect or given from previous iterations
    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {
        if( mode == GC_INIT_WITH_RECT )
            initMaskWithRect( mask, img.size(), rect );
        else // flag == GC_INIT_WITH_MASK
            checkMask( img, mask );
        initGMMs( img, mask, bgdGMM, fgdGMM );
    }

	// Standard stuff, if nothing to do just end the program
    if( iterCount <= 0)
        return;

	// Check mask for any errors
    if( mode == GC_EVAL )
        checkMask( img, mask );

	// Simple parameters of our algorythm, used for setting up edge flows
    const double gamma = 50; // Gamma seems to be just a parameter for lambda, here 50
    const double lambda = 9*gamma; // Lambda is simply a max value for flow, be it from source or to target, here 450
    const double beta = calcBeta( img ); // Beta is a parameter, here 1/(2*avg(sqr(||color[i] - color[j]||)))
										 // 1 / 2*average distance in colors between neighbours

	// NWeights, the flow capacity of our edges
    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );

	// The main loop
    for( int i = 0; i < iterCount; i++ )
    {
		// Simply initialize the graph we will be using throughout the algorythm. It is created empty
        GCGraph<double> graph;

		// Check the image at mask, and depending if it's FGD or BGD, return the number of component that suits it most.
		// Answer (component numbers) is stored in compIdxs, it does not store anything else
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );

		// This one adds samples to proper GMMs based on best component
        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );

		// NOTE: As far as I can tell these two will be primarily worked upon
		// Construct grapg cut graph, including initializing s and t values for source and sink flows
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );

		// Using max flow algorythm calculate segmentation
        estimateSegmentation( graph, mask );
    }
}
