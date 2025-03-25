1.此仓库用来记录结合isce2、ENVI、实现Polinsar配准后Range filter的步骤；   
2.其中Look_angle.py 计算视角->《The Wavenumber Shift in SAR Interferometry》中的off-nadir angle。代码中由于卫星处于变轨/升轨，因此对卫星轨道高度进行插值，以保证每行像元的卫星轨道高度参数一致；   
3.slope_angle.py 用于结合isce2结果中z.rdr.full文件计算坡度；   
4.range_filter用于结合上两步中结果，计算range filter结果并保存为envi格式；   
5：make_T6 是用来计算T6矩阵，以及复相干优化  
注：SAOCOM卫星中头文件系统带宽需要自己计算(我这里利用了阿根廷一位Dr.Santiago提供的插件，https://github.com/gmtsar/user-contributions/tree/main/saocom_slc)生成卫星参数文件计算得到(建议还是结合gpt阅读参数并计算系统带宽)；   
抱怨一下：这卫星数据头文件有点抽象(或也许是我经验不足)，在处理的过程中也没有找到相关教程，过程极其艰难........
