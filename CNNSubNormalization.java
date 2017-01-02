package deeplearning;

import java.io.*;

/**
 * 
 * 畳み込みニューラルネットワーク：畳み込み層
 *
 */

public class CNNSubNormalization extends CNNBaseLayer{

	int sizeF;	//フィルタサイズ
	final int filterAreaSize;	//局所フィルタ内ノード数
	
	//sizeNin:入力ノード数 iSizeX:入力データ横サイズ iSizeY:入力データ縦サイズ
	public CNNSubNormalization(int sizeNin , int iSizeX , int iSizeY, int sizeF){
			this.layerCode = 1052;
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
		float subNormSum;
		float[][][] output = new float[sizeNout][oSizeX][oSizeY];
		final int filterStart = sizeF/(-2);
		final int filterEnd = sizeF/2;
		
		//正規化
		for(i = 0; i < iSizeX; i++){
			for(j = 0; j < iSizeY; j++){
				//合計値を算出
				subNormSum = 0;
				for(n = 0; n < sizeNin; n++){
					for(k = filterStart; k <= filterEnd; k++){
						for(l = filterStart; l <= filterEnd; l++){
							filterX = i+k;
							filterY = j+l;
							if(filterX >= 0 && filterY >= 0 && filterX < iSizeX && filterY < iSizeY){
								subNormSum += input[n][filterX][filterY]/filterAreaSize;
							}
						}
					}
				}
				//正規化処理
				for(n = 0; n < sizeNin; n++){
					output[n][i][j] = input[n][i][j] - subNormSum;
				}
			}
		}
		return output;
	}

	public float[][][] backPropagation(float[][][] error) {
		int ni, no, i, j, k, l, filterX, filterY;
		float subDif;
		float[][][] nextError = new float[sizeNin][iSizeX][iSizeY];
		final int filterStart = sizeF/(-2);
		final int filterEnd = sizeF/2;

		for(ni = 0; ni < sizeNin; ni++){
			for(i = 0; i < iSizeX; i++){
				for(j = 0; j < iSizeY; j++){
					for(no = 0; no < sizeNout; no++){
						for(k = filterStart; k <= filterEnd; k++){
							for(l = filterStart; l <= filterEnd; l++){
								filterX = i+k;
								filterY = j+l;
								if(filterX >= 0 && filterY >= 0 && filterX < iSizeX && filterY < iSizeY){
									subDif = 0;
									if(k==0 && l==0 && ni==no){	//入力と出力が全く同じ座標かつ中心の場合
										subDif = 1.0f;
									}
									subDif -= 1.0f/filterAreaSize; 
									nextError[ni][i][j] += subDif*error[no][filterX][filterY];
								}
							}
						}
					}
				}
			}
		}
		return nextError;
	}

	public void training() {
	}

	public void save(BufferedWriter bw) throws IOException {
	}

	public void load(BufferedReader br) throws IOException {
	}
	
}