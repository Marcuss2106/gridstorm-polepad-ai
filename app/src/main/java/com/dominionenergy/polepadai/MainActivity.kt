package com.dominionenergy.polepadai

import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.dominionenergy.polepadai.ui.CameraUI
import com.dominionenergy.polepadai.ui.Info
import com.dominionenergy.polepadai.ui.ScanGallery
import com.dominionenergy.polepadai.ui.FormVerification
import com.dominionenergy.polepadai.ui.Upload
import com.dominionenergy.polepadai.ui.theme.DominionBlue
import com.dominionenergy.polepadai.ui.theme.DominionEnergyPolepadAITheme


// reworking -- homescreen is composable, mainactivity focuses more on the navigation between the screens
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            DominionEnergyPolepadAITheme {
                val navController = rememberNavController() //let's us start to navigate
                //val capturedImage = remember { mutableStateOf<Bitmap?>(null)} // hold the information about the camera?
                NavHost(
                    navController = navController,
                    startDestination = "home"
                ) {
                    composable("home") {
                        HomeScreen(navController)
                    }
                    composable("camera") { CameraUI() } // go to camera
                    composable("gallery") { ScanGallery() } // go to the gallery
                    composable("upload") { Upload() } // go to the uploaf screen
                    composable("info") { Info() } //go to info
                    composable("form") { FormVerificationScreenUI() } // go to form verification screen
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(navController: NavController) { // home screen composable + UI + nav'ing

    Scaffold(
        topBar = {
            Box(
                modifier = Modifier
                    .background(DominionBlue)
                    .fillMaxWidth()
                    .padding(start = 12.dp, top = 20.dp, end = 0.dp, bottom = 0.dp) // rememebr this for scrollability
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically
                ) {
                    Image(
                        painter = painterResource(R.drawable.backarrow), // back arrow
                        contentScale = ContentScale.Crop,
                        contentDescription = "Back Arrow",
                        modifier = Modifier
                            .size(40.dp)
                            // can't go back anywhere this is the homescreen
                    )

                    Spacer(modifier = Modifier.weight(1f))

                    Image(
                        painter = painterResource(R.drawable.dominionlogo),
                        contentScale = ContentScale.Crop,
                        contentDescription = "Logo",
                        modifier = Modifier
                            .size(100.dp)
                            .clickable {
                                navController.navigate("form")
                            }
                    )
                }
            }
        },

        bottomBar = {
            Box(
                modifier = Modifier
                    .background(DominionBlue)
                    .fillMaxWidth()
                    .padding(start = 5.dp, top = 7.5.dp, end = 0.dp, bottom = 0.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 15.dp, vertical = 20.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center
                ) {
                    Image(
                        painter = painterResource(R.drawable.galleryicon), // go to the gallery
                        contentScale = ContentScale.Crop,
                        contentDescription = "Gallery",
                        modifier = Modifier
                            .size(75.dp)
                            .clickable {
                                navController.navigate("gallery")
                            }
                    )

                    Spacer(modifier = Modifier.weight(1f))

                    Image(
                        painter = painterResource(R.drawable.camicon), // go to cameras
                        contentScale = ContentScale.Crop,
                        contentDescription = "Camera",
                        modifier = Modifier
                            .size(75.dp)
                            .clickable {
                                navController.navigate("camera")
                            }
                    )

                    Spacer(modifier = Modifier.weight(1f))

                    Image(
                        painter = painterResource(R.drawable.uploadarrow), // go to uploads
                        contentScale = ContentScale.Crop,
                        contentDescription = "Upload",
                        modifier = Modifier
                            .size(90.dp)
                            .clickable {
                                navController.navigate("upload")
                            }
                    )
                }
            }
        }

    ) { innerPadding ->
        // meat of home screen sandwich
        Box(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize(),
                contentAlignment = Alignment.Center
        ) {
            Text("Let's get snapping!", fontSize = 20.sp)
        }
    }
}
