package deeplearning;

import java.util.*;

/**
 * 
 * 畳み込みニューラルネットワーク：畳み込み層
 *
 */

public class CNNConvolution extends CNNBaseLayerConvolution{
	public boolean topLayer;			//先頭レイヤーの場合はtrue
	
	private float[][][] beforeError;	//直前の誤差
	
	CNNConvolution(int sizeN, int sizeX, int sizeY, int sizeF, int sizeNout, boolean isTop){
		//畳み込み層の構造の設定
		this.layerCode = 1011;
		this.sizeNin = sizeN;
		this.iSizeX = sizeX;
		this.iSizeY = sizeY;
		this.sizeF = sizeF;
		this.sizeNout = sizeNout;
		this.topLayer = isTop;
		
		this.oSizeX = iSizeX - sizeF + 1;
		this.oSizeY = iSizeY - sizeF + 1;
		
		//乱数生成器
		Random rnd = new Random();
		
		//初期化処理
		filter = new float[sizeNin][sizeNout][sizeF][sizeF];
		bias = new float[sizeNout];
		for(int n = 0 ; n < sizeNin ; n++){
			for( int kernel = 0 ; kernel < sizeNout ; kernel++){
				for( int i = 0 ; i < sizeF ; i++){
					for( int j = 0 ; j < sizeF ; j++){
						filter[n][kernel][i][j] = rnd.nextFloat();
					}
				}
			}
		}for( int kernel = 0 ; kernel < sizeNout ; kernel++){
			bias[kernel] = 1.0f;
		}
		gradient = new float[sizeNin][sizeNout][sizeF][sizeF];
		gradient_bias = new float[sizeNout];
		//lambda = 0.00001f;
		
		learnRate = 0.01f;
	}
	
	//学習率設定
	public void setLearningRate(float f){
		if(f < 0) f = 0.0f;
		else if(f > 1.0f) f = 1.0f;
		learnRate = f;
	}
	
	//順伝播
	//引数：入力データ　返り値：出力データ
	int a = 0;
	public float[][][] propagation(float[][][] input){
		int kernel, n, i, j, k, l;
		float max[] = new float[sizeNout];
		//float min[] = new float[sizeNout];
		float[] sum = new float[sizeNout];
		float[][][] output = new float[sizeNout][oSizeX][oSizeY];
		
		//最初のレイヤーの場合（カラーチャンネルがあるのでフィルタの計算方法が変化）
		for( kernel = 0 ; kernel < sizeNout ; kernel++){
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					output[kernel][i][j] = 0;
					for( n = 0 ; n < sizeNin ; n++){
						for( k = 0 ; k < sizeF ; k++){
							for( l = 0 ; l < sizeF ; l++){
								output[kernel][i][j] += input[n][i+k][j+l] * filter[n][kernel][k][l];
							}
						}
					}
					output[kernel][i][j] += bias[kernel];
					sum[kernel] += output[kernel][i][j];
					if(max[kernel] < output[kernel][i][j]) max[kernel] = output[kernel][i][j];
				}
			}
		}
		
		//正規化
		/*for( kernel = 0 ; kernel < sizeNout ; kernel++){
			sum[kernel] /= oSizeX*oSizeY;
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					output[kernel][i][j] -= sum[kernel]; 
				}
			}
		}*/
		
		//if(sizeNin == 8){
			//print(output);
			//if(a++ == 3) System.exit(0);
		//}
		
		/*if(sizeNin == 8){
			for( kernel = 0 ; kernel < sizeNout ; kernel++){
				System.out.println(kernel+":"+output[kernel][0][0]);
			}
		}*/
		
		//学習用に入力を記憶
		if(!useMinibatch){
			beforeInput = input;
		}
		return output;
	}

	//逆伝播
	//引数:error[ノード番号][入力X軸][入力Y軸] 前の層の誤差　input[ノード番号][入力X軸][入力Y軸]　この層に対する入力
	public float[][][] backPropagation_(float[][][] error){
		int kernel, n, i, j, k, l;
		float[][][] nextError = new float[sizeNin][iSizeX][iSizeY];
		float sum;
		final float rmsprop_alpha2 = 1 - rmsprop_alpha; 

		//重みの逆伝播
		if(!topLayer){
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					for( kernel = 0 ; kernel < sizeNout ; kernel++){
						if(Math.abs(error[kernel][i][j]) > 0.0001f){	//誤差が十分に小さいところは計算をしないことにより処理速度向上
							for( n = 0 ; n < sizeNin ; n++){
								for( k = 0 ; k < sizeF ; k++){
									for( l = 0 ; l < sizeF ; l++){
										nextError[n][i+k][j+l] += error[kernel][i][j] * filter[n][kernel][k][l];
									}
								}
								//System.out.println("NEXT ERR CONV:"+sum + " i:"+i+" j:"+j);
							}
						}
					}
				}
			}
		}
		//重みの学習のための更新値を算出
		for( n = 0; n < sizeNin; n++){
			for( kernel = 0 ; kernel < sizeNout ; kernel++){
				for( k = 0 ; k < sizeF ; k++){
					for( l = 0 ; l < sizeF ; l++){
						sum = 0;
						for( i = 0 ; i < oSizeX ; i++){
							for( j = 0 ; j < oSizeY ; j++){
								sum += error[kernel][i][j] * beforeInput[n][i+k][j+l];
							}
						}
						gradient[n][kernel][k][l] = rmsprop_alpha * gradient[n][kernel][k][l] + rmsprop_alpha2*sum*sum;
						filter[n][kernel][k][l] -= learnRate*sum/(Math.sqrt(gradient[n][kernel][k][l])+eps);
					}
				}
			}
		}
		//バイアスの学習
		for( kernel = 0 ; kernel < sizeNout ; kernel++){
			sum = 0;
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					sum += error[kernel][i][j];
				}
			}
			gradient_bias[kernel] += rmsprop_alpha	* gradient_bias[kernel]+ rmsprop_alpha2 * sum * sum;
			bias[kernel] -= sum*learnRate / (Math.sqrt(gradient_bias[kernel]) + eps);
		}
		return nextError;
	}

	//逆伝播(ミニバッチ用)
	//引数:error[ノード番号][入力X軸][入力Y軸] 前の層の誤差　input[ノード番号][入力X軸][入力Y軸]　この層に対する入力
	public float[][][] backPropagation(float[][][] error){
		int kernel, n, i, j, k, l;
		float sum;
		final float rmsprop_alpha2 = 1 - rmsprop_alpha; 
		float[][][] nextError = new float[sizeNin][iSizeX][iSizeY];

		//重みの逆伝播
		if(!topLayer){
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					for( kernel = 0 ; kernel < sizeNout ; kernel++){
						if(Math.abs(error[kernel][i][j]) > 0.0001f){	//誤差が十分に小さいところは計算をしないことにより処理速度向上
							for( n = 0 ; n < sizeNin ; n++){
								for( k = 0 ; k < sizeF ; k++){
									for( l = 0 ; l < sizeF ; l++){
										nextError[n][i+k][j+l] += error[kernel][i][j] * filter[n][kernel][k][l];
										if(Float.isNaN(nextError[n][i+k][j+l])) System.out.println("CONV ERR NAN:"+n+":"+(i+k)+":"+(j+l)+":"+kernel);
									}
								}
								//System.out.println("NEXT ERR CONV:"+nextError[n][i][j] + " i:"+i+" j:"+j);
							}
						}
					}
				}
			}
		}
		//バイアスの学習
		for( kernel = 0 ; kernel < sizeNout ; kernel++){
			//System.out.println(kernel+":"+filter[0][kernel][0][0]);
			sum = 0;
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					sum += error[kernel][i][j];
				}
			}
			gradient_bias[kernel] += rmsprop_alpha	* gradient_bias[kernel]+ rmsprop_alpha2 * sum * sum;
			bias[kernel] -= sum*learnRate / (Math.sqrt(gradient_bias[kernel]) + eps);
		}
		if(Float.isNaN(nextError[0][0][0])) System.out.println("CONV1 EXCEPTION!!");
		if(Float.isNaN(error[0][0][0])) System.out.println("CONV2 EXCEPTION!!");
		this.beforeError = error;
		return nextError;
	}
	
	//学習
	public void training(){
		int kernel, n, i, j, k, l;
		float sum;
		final float rmsprop_alpha2 = 1 - rmsprop_alpha; 
		//重みの学習のための更新値を算出
		for( n = 0; n < sizeNin; n++){
			for( kernel = 0 ; kernel < sizeNout ; kernel++){
				for( k = 0 ; k < sizeF ; k++){
					for( l = 0 ; l < sizeF ; l++){
						sum = 0;
						for( i = 0 ; i < oSizeX ; i++){
							for( j = 0 ; j < oSizeY ; j++){
								sum += beforeError[kernel][i][j] * beforeInput[n][i+k][j+l];
								//if(Float.isNaN(sum)) System.out.println("CONV NAN:"+kernel+":"+i+":"+j);
							}
						}
						gradient[n][kernel][k][l] = rmsprop_alpha * gradient[n][kernel][k][l] + rmsprop_alpha2*sum*sum;
						filter[n][kernel][k][l] -= learnRate*sum/(Math.sqrt(gradient[n][kernel][k][l])+eps);
						//if(Float.isNaN(sum)) System.out.println("CONV NAN:"+n+":"+kernel+":"+k+":"+l);
					}
				}
			}
		}
		//System.out.println(filter[0][0][0][0]);
	}
	
	//出力
	public void print(float[][][] output){
		for(int n = 0; n < sizeNout; n++){
			System.out.println();
			System.out.println("畳み込み層["+n+"] 入力"+iSizeX);
			for(int j = 0 ; j < oSizeX ; j++){
				for(int i = 0 ; i < oSizeY ; i++){
					System.out.print(String.format("%2.1f",output[n][j][i]) + "  ");
					/*if(output[0][i][j] > 2.0f) System.out.print("■");
					else if(output[0][i][j] > 1.2f) System.out.print("●");
					else if(output[0][i][j] > 0.8f) System.out.print("◎");
					else if(output[0][i][j] > 0.4f) System.out.print("□");
					else if(output[0][i][j] > 0.0f) System.out.print("・");
					else System.out.print("　");*/
				}
				System.out.println();
			}
		}
	}
	
	//テスト用
	public static void main(String[] args){
		float[][][] image = {{{0.9f , 0.0f , 0.0f , 0.0f , 0.0f , 0.0f , 0.0f , 0.1f , 0.9f},
							{0.9f , 0.1f , 0.0f , 0.1f , 0.0f , 0.0f , 0.9f , 0.9f , 0.9f},
							{0.9f , 0.9f , 0.1f , 0.0f , 0.0f , 0.1f , 0.9f , 0.1f , 0.0f},
							{0.1f , 0.9f , 0.1f , 0.1f , 0.0f , 0.1f , 0.9f , 0.1f , 0.0f},
							{0.1f , 0.9f , 0.9f , 0.1f , 0.0f , 0.9f , 0.9f , 0.9f , 0.1f},
							{0.0f , 0.1f , 0.9f , 0.9f , 0.9f , 0.1f , 0.2f , 0.9f , 0.1f}}};
		System.out.println("畳み込み処理前");
		for(int j = 0 ; j < 6 ; j++){
			for(int i = 0 ; i < 9 ; i++){
				System.out.print(image[0][j][i] + " ");
			}
			System.out.println();
		}
		System.out.println("畳み込み処理後");
		CNNConvolution conv = new CNNConvolution(1,6,9,3,1,true);
		float[][][] output = conv.propagation(image);
		for(int j = 0 ; j < conv.oSizeX ; j++){
			for(int i = 0 ; i < conv.oSizeY ; i++){
				System.out.print(String.format("%.2f",output[0][j][i]) + " ");
			}
			System.out.println();
		}
	}
}