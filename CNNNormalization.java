package deeplearning;

import java.io.*;

/**
 * 
 * 畳み込みニューラルネットワーク：畳み込み層
 *
 */

public class CNNNormalization extends CNNBaseLayer{

	int sizeF;	//フィルタサイズ
	final int filterAreaSize;	//局所フィルタ内ノード数
	final float divNormRange = 0.001f;
	
	//sizeNin:入力ノード数 iSizeX:入力データ横サイズ iSizeY:入力データ縦サイズ
	public CNNNormalization(int sizeNin , int iSizeX , int iSizeY, int sizeF){
			this.layerCode = 1051;
			this.sizeNin = sizeNin;
			this.sizeNout = sizeNin;
			this.iSizeX = iSizeX;
			this.iSizeY = iSizeY;
			this.oSizeX = iSizeX;
			this.oSizeY = iSizeY;
			this.sizeF = sizeF;
			if(sizeF%2 == 0){
				this.sizeF += 1;	//フィルタサイズを奇数にする
			}
			filterAreaSize = sizeNin*this.sizeF*this.sizeF;
	}
	
	public float[][][] propagation(float[][][] input) {
		int n, i, j, k, l, filterX, filterY;
		float subNormSum, divNormSum;
		float[][][] output = new float[sizeNout][oSizeX][oSizeY];
		final int filterStart = sizeF/(-2);
		final int filterEnd = sizeF/2;
		
		//正規化
		for(i = 0; i < iSizeX; i++){
			for(j = 0; j < iSizeY; j++){
				//合計値を算出
				subNormSum = 0;
				divNormSum = 0;
				for(n = 0; n < sizeNin; n++){
					for(k = filterStart; k <= filterEnd; k++){
						for(l = filterStart; l <= filterEnd; l++){
							filterX = i+k;
							filterY = j+l;
							if(filterX >= 0 && filterY >= 0 && filterX < iSizeX && filterY < iSizeY){
								subNormSum += input[n][filterX][filterY]/filterAreaSize;
								divNormSum += input[n][filterX][filterY]*input[n][filterX][filterY]/filterAreaSize;
							}
						}
					}
				}
				//除算正規化用
				divNormSum = (float)Math.sqrt((double)divNormSum);
				if(divNormSum < divNormRange){
					divNormSum = divNormRange;	//値の下限を設定
				}
				//正規化処理
				for(n = 0; n < sizeNin; n++){
					output[n][i][j] = (input[n][i][j] - subNormSum)/divNormSum;
				}
			}
		}
		if(!useMinibatch) beforeInput = input;
		return output;
	}

	public float[][][] backPropagation(float[][][] error) {
		int ni, no, i, j, k, l, filterX, filterY;
		float subDif, divDif;
		float[][] divNormSum = new float[iSizeX][iSizeY];
		float[][][] nextError = new float[sizeNin][iSizeX][iSizeY];
		final int filterStart = sizeF/(-2);
		final int filterEnd = sizeF/2;

		for(i = 0; i < iSizeX; i++){
			for(j = 0; j < iSizeY; j++){
				//合計値を算出
				for(ni = 0; ni < sizeNin; ni++){
					for(k = filterStart; k <= filterEnd; k++){
						for(l = filterStart; l <= filterEnd; l++){
							filterX = i+k;
							filterY = j+l;
							if(filterX >= 0 && filterY >= 0 && filterX < iSizeX && filterY < iSizeY){
								divNormSum[i][j] += beforeInput[ni][filterX][filterY]*beforeInput[ni][filterX][filterY]/filterAreaSize;
							}
						}
					}
				}
				divNormSum[i][j] = (float)Math.sqrt((double)divNormSum[i][j]);
				if(divNormSum[i][j] < divNormRange){
					divNormSum[i][j] = divNormRange;	//値の下限を設定
				}
			}
		}

		for(ni = 0; ni < sizeNin; ni++){
			for(i = 0; i < iSizeX; i++){
				for(j = 0; j < iSizeY; j++){
					for(no = 0; no < sizeNout; no++){
						for(k = filterStart; k <= filterEnd; k++){
							for(l = filterStart; l <= filterEnd; l++){
								filterX = i+k;
								filterY = j+l;
								if(filterX >= 0 && filterY >= 0 && filterX < iSizeX && filterY < iSizeY){
									divDif = 0;
									subDif = 0;
									if(k==0 && l==0 && ni==no){	//入力と出力が全く同じ座標かつ中心の場合
										divDif = 1.0f/divNormSum[filterX][filterY];
										subDif = 1.0f;
									}
									divDif -= beforeInput[ni][i][j]*beforeInput[ni][i][j]/
											(filterAreaSize*divNormSum[filterX][filterY]*divNormSum[filterX][filterY]*divNormSum[filterX][filterY]);
									subDif -= 1.0f/filterAreaSize; 
									nextError[ni][i][j] += subDif*divDif*error[no][filterX][filterY];
								}
							}
						}
					}
					//System.out.println(nextError[ni][i][j]);
				}
			}
		}
		if(Float.isNaN(nextError[0][0][0])) System.out.println("NORM1 EXCEPTION!!");
		if(Float.isNaN(error[0][0][0])) System.out.println("NORM2 EXCEPTION!!");
		//System.exit(0);
		return nextError;
	}

	public void training() {
	}

	public void save(BufferedWriter bw) throws IOException {
	}

	public void load(BufferedReader br) throws IOException {
	}
	
}