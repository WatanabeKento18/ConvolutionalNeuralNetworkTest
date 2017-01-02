package deeplearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;

/**
 * 
 * 畳み込みニューラルネットワーク：プーリング層
 *
 */

public class CNNPooling extends CNNBaseLayer{
	boolean poolingMap[][][];		//プーリングにより採用された座標を保存
	
	CNNPooling(int sizeN , int sizeX , int sizeY){
		this.layerCode = 1031;
		this.sizeNin = this.sizeNout = sizeN;
		this.iSizeX = sizeX;
		this.iSizeY = sizeY;
		this.oSizeX = (iSizeX+1)/2;
		this.oSizeY = (iSizeY+1)/2;
		
		poolingMap = new boolean[sizeN][sizeX][sizeY];		
	}
	
	//順伝播(Max Pooling)
	//引数:[ノード番号][入力X軸][入力Y軸]
	public float[][][] propagation(float[][][] input){
		int n , i , j , k , l , fx , fy;
		int maxx = 0 , maxy = 0;
		float[][][] output = new float[sizeNin][oSizeX][oSizeY];
		float max;
		for( n = 0 ; n < sizeNin ; n++){
			for( i = 0 ; i < oSizeX ; i++){
				for( j = 0 ; j < oSizeY ; j++){
					max = 0;
					maxx = -1;
					for( k = 0 ; k < 2 ; k++){
						for( l = 0 ; l < 2 ; l++){
							fx = i*2+k;
							fy = j*2+l;
							if(fx >= iSizeX || fy >= iSizeY) continue;
							poolingMap[n][fx][fy] = false;
							if(max < input[n][fx][fy]){
								max = input[n][fx][fy];
								maxx = fx;
								maxy = fy;
							}
						}
						output[n][i][j] = max;
						if(maxx != -1) poolingMap[n][maxx][maxy] = true;
					}
				}
			}
		}
		//print(output);
		return output;
	}
	
	//逆伝播
	//引数:[ノード番号][入力X軸][入力Y軸]
	public float[][][] backPropagation(float[][][] error){
		int n , i , j;
		float[][][] nextError = new float[sizeNin][iSizeX][iSizeY];
		for( n = 0 ; n < sizeNin ; n++){
			for( i = 0 ; i < iSizeX ; i++){
				for( j = 0 ; j < iSizeY ; j++){
					if(poolingMap[n][i][j]){
						nextError[n][i][j] = error[n][i/2][j/2];
					}else{
						nextError[n][i][j] = 0;
					}
					//System.out.println("NEXT ERR POOL:"+nextError[n][i][j] + " i:"+i+" j:"+j);
				}
			}
		}
		return nextError;
	}
	
	public void training(){
		
	}

	//出力
	public void print(float[][][] output){
		System.out.println("プーリング層 入力"+iSizeX);
		for(int j = 0 ; j < oSizeX ; j++){
			for(int i = 0 ; i < oSizeY ; i++){
				System.out.print(String.format("%.2f",output[1][j][i]) + " ");
			}
			System.out.println();
		}
	}
	
	//層情報の保存(CNNクラスのsaveメソッドより呼び出し)
	public void save(BufferedWriter bw) throws IOException{
		//プーリング層の基本情報を書き込み
		bw.write("1031");	//プーリング層コード
		bw.newLine();
		bw.write(String.format("%d", sizeNin));
		bw.newLine();
		bw.write(String.format("%d", iSizeX));
		bw.newLine();
		bw.write(String.format("%d", iSizeY));
		bw.newLine();
	}
	
	//層情報の読み込み(CNNクラスのloadメソッドより呼び出し)
	public void load(BufferedReader br) throws IOException{
	
	}
	
	//テスト用
	public static void main(String[] args){
		float[][][] image = {{	{-0.5f , 0.0f , 0.0f , 0.0f , 0.0f , 0.0f , 0.0f , 0.1f , 0.9f},
							{0.9f , 0.5f , 0.0f , 0.1f , 0.0f , 0.0f , 0.9f , 0.9f , 0.7f},
							{-0.9f , 0.9f , 0.1f , 0.0f , 0.0f , 0.1f , 0.9f , 0.1f , 0.0f},
							{0.1f , 0.9f , 0.1f , -0.1f , 0.0f , 0.1f , 0.9f , 0.1f , 0.0f},
							{-0.1f , 0.9f , 0.9f , 0.1f , 0.0f , 0.9f , 0.9f , 0.9f , 0.0f},
							{0.0f , -0.1f , 0.9f , -0.9f , 0.9f , 0.1f , 0.2f , 0.9f , 0.1f}}};
		System.out.println("プーリング処理前");
		for(int j = 0 ; j < 6 ; j++){
			for(int i = 0 ; i < 9 ; i++){
				System.out.print(image[0][j][i] + " ");
			}
			System.out.println();
		}
		System.out.println("プーリング処理後");
		CNNPooling pool = new CNNPooling(1,6,9);
		float[][][] output = pool.propagation(image);
		for(int i = 0 ; i < output[0].length ; i++){
			for(int j = 0 ; j < output[0][0].length ; j++){
				System.out.print(String.format("%.1f",output[0][i][j]) + " ");
			}
			System.out.println();
		}
	}
}