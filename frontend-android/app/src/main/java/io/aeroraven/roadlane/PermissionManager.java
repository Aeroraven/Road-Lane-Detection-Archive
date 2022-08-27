package io.aeroraven.roadlane;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Build;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class PermissionManager {
    public static final int REQUEST_CODE = 5;
    private static final String[] permission = new String[]{
            Manifest.permission.INTERNET,
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
    };
    public static boolean isPermissionGranted(Activity activity){
        if(Build.VERSION.SDK_INT >= 23){
            for(int i = 0; i < permission.length;i++) {
                int checkPermission = ContextCompat.checkSelfPermission(activity,permission[i]);
                if(checkPermission != PackageManager.PERMISSION_GRANTED){
                    return false;
                }
            }
            return true;
        }else{
            return true;
        }
    }

    public static boolean checkPermission(Activity activity){
        if(isPermissionGranted(activity)) {
            return true;
        } else {
            //如果没有设置过权限许可，则弹出系统的授权窗口
            ActivityCompat.requestPermissions(activity,permission,REQUEST_CODE);
            return false;
        }
    }
}
