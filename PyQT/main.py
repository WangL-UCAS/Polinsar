import sys
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PyQt5.QtWidgets import QProgressBar
from Forest_ui import Ui_GSVInversionClass  # 自动生成的 UI 文件
import os
from PyQt5.QtCore import QPropertyAnimation, QPoint
from PyQt5.QtGui import QPainter, QLinearGradient, QColor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_GSVInversionClass()
        self.ui.setupUi(self)

        # === 按钮绑定 ===
        self.ui.pushButton_TTM.clicked.connect(self.choose_TTM_file)
        self.ui.pushButton_OtherData.clicked.connect(self.choose_other_file)
        self.ui.pushButton_OutPath.clicked.connect(self.choose_out_file)
        self.ui.OK.clicked.connect(self.run_inversion)
        self.ui.Cancel.clicked.connect(self.close)
        # 设置进度范围
        self.ui.progressBar.setRange(0, 19)
        print("GSV Inversion Tool v1.0 启动完成")
    def choose_TTM_file(self):
        '''
        读取 t11 t22 om的bin文件，并返回合并的路径信息
        :return:合并后的路径信息
        '''
        # 选择文件夹
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含 T11、T22、Om 文件的文件夹", "E:/")
        if folder_path:
            # 遍历文件夹下的所有 bin 文件
            bin_files = [f for f in os.listdir(folder_path) if f.endswith(".bin")]

            # 初始化路径
            t11_path = t22_path = om_path = ""

            for f in bin_files:
                f_upper = f.upper()
                full_path = os.path.normpath(os.path.join(folder_path, f))  # 标准化路径
                if "T11" in f_upper:
                    t11_path = full_path
                elif "T22" in f_upper:
                    t22_path = full_path
                elif "OM" in f_upper:
                    om_path = full_path

            # 检查是否找全了
            missing = []
            if not t11_path:
                missing.append("T11")
            if not t22_path:
                missing.append("T22")
            if not om_path:
                missing.append("Om")

            if missing:
                # 弹出提示框
                missing_msgs = [f"缺少{item}文件" for item in missing]
                QMessageBox.warning(
                    self,
                    "文件缺失",
                    f"以下文件缺失：{', '.join(missing_msgs)}. 请确认文件夹中包含所需的 .bin 文件。"
                )
                return

            # 合并路径并显示
            combined_path = f"{t11_path};{t22_path};{om_path}"
            self.ui.T11T22Om.setText(combined_path)


    # 选取坡度、入射角、树种、经纬度、kz路径
    def choose_other_file(self):
        '''
        读取 坡度、入射角、树种、经纬度、kz路径
        :return: 合并的路径信息
        '''
        # 选择文件夹
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含 slope、angle、tree、lon/lat、kz 的 TIF 文件的文件夹",
                                                       "E:/")
        if folder_path:
            # 遍历文件夹下所有 tif 文件（忽略大小写）
            tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tif")]

            # 初始化路径变量
            slope_path = angle_path = tree_path = lonlat_path = kz_path = ""

            for f in tif_files:
                fname_lower = f.lower()
                full_path = os.path.normpath(os.path.join(folder_path, f)).replace("\\", "/")  # 标准化并统一斜杠
                if "slope" in fname_lower:
                    slope_path = full_path
                elif "angle" in fname_lower:
                    angle_path = full_path
                elif "tree" in fname_lower:
                    tree_path = full_path
                elif "lon" in fname_lower or "lat" in fname_lower:
                    lonlat_path = full_path
                elif "kz" in fname_lower:
                    kz_path = full_path

            # 检查缺失项
            missing = []
            if not slope_path:
                missing.append("slope")
            if not angle_path:
                missing.append("angle")
            if not tree_path:
                missing.append("tree")
            if not lonlat_path:
                missing.append("lon/lat")
            if not kz_path:
                missing.append("kz")

            if missing:
                missing_msgs = [f"缺少{item}文件" for item in missing]
                QMessageBox.warning(
                    self,
                    "文件缺失",
                    f"以下文件缺失：{', '.join(missing_msgs)}. 请确认文件夹中包含所需的 TIF 文件。"
                )
                return

            # 拼接路径字符串（你可以按需要调整顺序或分隔符）
            combined_path = f"{slope_path};{angle_path};{tree_path};{lonlat_path};{kz_path}"
            self.ui.OtherData.setText(combined_path)


    #输出路径
    def choose_out_file(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择输出文件夹", "E:/")
        if folder_path:
            normalized_path = os.path.normpath(folder_path).replace("\\", "/")
            self.ui.OutPath.setText(normalized_path)
    def run_inversion(self):
        #判断逻辑，是否为空
        TTM_text = self.ui.T11T22Om.text()
        TTM_path = TTM_text.split(";") if TTM_text else []
        if len(TTM_path) < 3:
            QMessageBox.warning(self, "输入错误", "请先选择 T11、T22 和 Om 文件夹，确保路径填写完整。")
            return

        OtherData_text = self.ui.OtherData.text()
        Other_Data_path = OtherData_text.split(";") if OtherData_text else []
        if len(Other_Data_path) < 5:
            QMessageBox.warning(self, "输入错误", "请先选择坡度、入射角、树种、经纬度和kz文件夹，确保路径填写完整。")
            return

        output_path = self.ui.OutPath.text()
        if not output_path:
            QMessageBox.warning(self, "输入错误", "请先选择输出文件夹路径。")
            return
        # === 获取输入 ===
        t11_path, t22_path, omega_path = TTM_path[0], TTM_path[1], TTM_path[2]

        slope_path, inc_angle_path, tree_species_path,  lon_lat_path, kz_path = Other_Data_path[0], Other_Data_path[1], Other_Data_path[2], Other_Data_path[3], Other_Data_path[4]

        output_path = self.ui.OutPath.text()

        # === 字符串字段检查 ===
        required_paths = {
            "T11": t11_path,
            "T22": t22_path,
            "干涉项（OM）": omega_path,
            "坡度": slope_path,
            "入射角": inc_angle_path,
            "树种": tree_species_path,
            "经纬度":lon_lat_path,
            "垂直波束（kz）":kz_path,
            "输出路径": output_path,

        }
        missing = [k for k, v in required_paths.items() if not v]
        if missing:
            QMessageBox.warning(self, "缺失输入", f"以下字段未填写：{', '.join(missing)}")
            return

        # === 调用反演模块 ===
        try:
            import run_gsv_inversion
            run_gsv_inversion.run_gsv_inversion(
                t11_path, t22_path, omega_path, slope_path, inc_angle_path, tree_species_path,
                lon_lat_path, kz_path, output_path,progressbar=self.ui.progressBar
            )
            QMessageBox.information(self, "保存成功", "蓄积量结果文件已保存。")
        except Exception as e:
            print(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"发生错误：\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("GSV 参数反演工具")
    window.show()
    sys.exit(app.exec_())
