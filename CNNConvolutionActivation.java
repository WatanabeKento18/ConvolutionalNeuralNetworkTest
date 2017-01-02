package deeplearning;

import java.util.Random;

/**
 * 
 * 畳み込みニューラルネットワーク：活性化畳み込み層
 *
 */

public class CNNConvolutionActivation{
	int sizeN;					//ノード数
	int iSizeX , iSizeY;		//入力サイズ
	int sizeF;					//フィルタサイズ
	int kernelNum;				//カーネル数
	public boolean topLayer;			//先頭レイヤーの場合はtrue
	
	private float[][][] filter;	//畳み込みフィルタ
	private float[] bias;		//バイアス
	
	boolean activationMap[][][];		//活性化により採用された座標を保存
	
	private float[][][] beforeInput;	//直前の入力
	
	float learnRate;
	
	CNNConvolutionActivation(int sizeN , int sizeX, int sizeY, int sizeF, int kernelNum){
		this.sizeN = sizeN;
		this.iSizeX = sizeX;
		this.iSizeY = sizeY;
		this.sizeF = sizeF;
		this.kernelNum = kernelNum;
		
		//乱数生成器
		Random rnd = new Random();
		
		//初期化処理
		filter = new float[kernelNum][sizeF][sizeF];
		bias = new float[kernelNum];
		for( int kernel = 0 ; kernel < kernelNum ; kernel++){
			for( int i = 0 ; i < sizeF ; i++){
				for( int j = 0 ; j < sizeF ; j++){
					filter[kernel][i][j] = rnd.nextFloat();
				}
			}
		}for( int kernel = 0 ; kernel < kernelNum ; kernel++){
			bias[kernel] = 1.0f;
		}
		
		learnRate = 0.01f;
	}
	
	//順伝播
	//引数：入力データ　返り値：出力データ
	float[][][] propagation(float[][][] input){
		int kernel , n , i , j , k , l;
		int oSizeX = getOutputSizeX();
		int oSizeY = getOutputSizeY();
		float max[] = new float[kernelNum];
		float min[] = new float[kernelNum];
		float[][][] output = new float[kernelNum][oSizeX][oSizeY];
		for( kernel = 0 ; kernel < kernelNum ; kernel++){
			max[kernel] = 0;
			min[kernel] = 0;
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					for( n = 0 ; n < sizeN ; n++){
						output[kernel][i][j] = 0;
						for( k = 0 ; k < sizeF ; k++){
							for( l = 0 ; l < sizeF ; l++){
								output[kernel][i][j] += input[n][i+k][j+l] * filter[kernel][k][l];
							}
						}
						output[kernel][i][j] += bias[kernel];
						//System.out.println("CONV SIZE:"+ sizeN + " i:"+i+" j:"+j + " DATA:"+output[kernel][i][j]);
					}
					if(max[kernel] < output[kernel][i][j]){
						max[kernel] = output[kernel][i][j];
					}else if(min[kernel] > output[kernel][i][j]){
						min[kernel] = output[kernel][i][j];
					}
				}
			}
		}
		for( kernel = 0 ; kernel < kernelNum ; kernel++){
			if(max[kernel] > 0.01f){
			//System.out.println("CONV SIZE:"+ sizeN + " MAX:"+max[kernel]);
				for( i = 0 ; i < oSizeX ; i++){
					for( j = 0 ; j < oSizeY ; j++){
						output[kernel][i][j] /= max[kernel];
						if(Float.isInfinite(output[kernel][i][j])){
							System.out.println("ERROR CONV SIZE:"+ sizeN + " DATA:"+output[kernel][i][j] + " MAX:" + max[kernel]);
							System.exit(0);
						}
					}
				}
			}else{
				float min_abs = min[kernel]*-1;
				for( i = 0 ; i < oSizeX ; i++){
					for( j = 0 ; j < oSizeY ; j++){
						output[kernel][i][j] = (output[kernel][i][j] + min_abs) / min_abs;
					}
				}
			}
		}
		//print(output);
		//学習用に入力を記憶
		beforeInput = input;
		return output;
	}

	//逆伝播
	//引数:error[ノード番号][入力X軸][入力Y軸] 前の層の誤差　input[ノード番号][入力X軸][入力Y軸]　この層に対する入力
	public float[][][] backPropagation(float[][][] error){
		int kernel , n , i , j , k , l;
		int oSizeX = getOutputSizeX();
		int oSizeY = getOutputSizeY();
		float[][][] nextError = new float[sizeN][iSizeX][iSizeY];
		float sum;
		//重みの学習
		for( n = 0 ; n < sizeN ; n++){
			for( k = 0 ; k < sizeF ; k++){
				for( l = 0 ; l < sizeF ; l++){
					for( kernel = 0 ; kernel < kernelNum ; kernel++){
						sum = 0;
						for( i = 0 ; i < oSizeX ; i++){
							for( j = 0 ; j < oSizeY ; j++){
								sum += error[kernel][i][j] * beforeInput[n][i+k][j+l];
							}
						}
						filter[kernel][k][l] -= sum*learnRate;
					}
				}
			}
		}
		//バイアスの学習
		for( kernel = 0 ; kernel < kernelNum ; kernel++){
			sum = 0;
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					sum += error[kernel][i][j];
				}
			}
			bias[kernel] -= sum*learnRate;
		}
		//重みの逆伝播
		if(!topLayer){
			for( n = 0 ; n < sizeN ; n++){
				for( i = 0 ; i < iSizeX ; i++){
					for( j = 0 ; j < iSizeY ; j++){
						sum = 0;
						nextError[n][i][j] = 0;
						for( kernel = 0 ; kernel < kernelNum ; kernel++){
							for( k = 0 ; k < sizeF ; k++){
								for( l = 0 ; l < sizeF ; l++){
									if(i-k < 0 || j-l < 0 || i-k >= oSizeX || j-l >= oSizeY) continue;
									sum += error[kernel][i-k][j-l] * filter[kernel][k][l];
								}
							}
						}
						nextError[n][i][j] = sum;
						//System.out.println("NEXT ERR CONV:"+sum + " i:"+i+" j:"+j);
					}
				}
			}
		}
		return nextError;
	}
	
	//出力層Xサイズを取得
	public int getOutputSizeX(){
		return iSizeX - sizeF + 1;
	}
	
	//出力層Yサイズを取得
	public int getOutputSizeY(){
		return iSizeY - sizeF + 1;
	}
	
	//出力
	public void print(float[][][] output){
		System.out.println("畳み込み層");
		for(int j = 0 ; j < getOutputSizeX() ; j++){
			for(int i = 0 ; i < getOutputSizeY() ; i++){
				System.out.print(String.format("%.2f",output[0][j][i]) + " ");
				/*if(output[0][i][j] > 0.2f) System.out.print("■");
				else System.out.print("□");*/
			}
			System.out.println();
		}
	}
	
}