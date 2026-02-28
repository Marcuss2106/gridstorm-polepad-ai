package com.dominionenergy.polepadai

import android.R.attr.end
import android.R.attr.top
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dominionenergy.polepadai.ui.theme.DominionBlue
import com.dominionenergy.polepadai.ui.theme.DominionEnergyPolepadAITheme

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DominionEnergyPolepadAITheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = {
                        Box(
                            modifier = Modifier
                                .background(DominionBlue)
                                .fillMaxWidth()
                                .padding(start = 12.dp,top = 20.dp, end = 0.dp, bottom = 0.dp)
                                    //removing the padding from the top
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 10.dp, vertical = 10.dp), //no left/right padding
                                     verticalAlignment = Alignment.CenterVertically
                            ) {
                                Image(painter = painterResource(id = R.drawable.backarrow),  contentScale = ContentScale.Crop, contentDescription = "Dominion Energy Logo", modifier = Modifier.size(40.dp))
                                Spacer(modifier = Modifier.weight(1f)) //fill available space, padding between arrow and logo
                                Image(painter = painterResource(id = R.drawable.dominionlogo), contentScale = ContentScale.Crop, contentDescription = "Dominion Energy Logo", modifier = Modifier.size(100.dp))
                            }
                        }
                    },
                    bottomBar = {
                        Box(
                            modifier = Modifier
                                .background(DominionBlue)
                                .fillMaxWidth()
                                .padding(start = 5.dp,top = 7.5.dp, end = 0.dp, bottom = 0.dp)
                               // .padding(top = WindowInsets.statusBars.asPaddingValues().calculateTopPadding())
                            //removing the padding from the top
                        ) {
                            Row(modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 15.dp, vertical = 20.dp), //no left/right padding
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.Center
                            )

                            {

                                Image(painter = painterResource(id = R.drawable.galleryicon),
                                    contentScale = ContentScale.Crop,
                                    contentDescription = "Dominion Energy Logo",
                                    modifier = Modifier.size(75.dp))
                                Spacer(modifier = Modifier.weight(1f))
                                Image(painter = painterResource(id = R.drawable.camicon), contentScale = ContentScale.Crop, contentDescription = "Dominion Energy Logo", modifier = Modifier.size(75.dp))
                                Spacer(modifier = Modifier.weight(1f))
                                Image(painter = painterResource(id = R.drawable.uploadarrow), contentScale = ContentScale.Crop, contentDescription = "Dominion Energy Logo", modifier = Modifier.size(90.dp))

                            }
                        }
                    }
                ) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier
                            .padding(innerPadding)
                            .fillMaxSize(),
                        contentAlignment = Alignment.Center
                    )
                }
            }
        }
    }
}

@Composable
fun Greeting(
    name: String,
    modifier: Modifier = Modifier,
    contentAlignment: Alignment = Alignment.Center
) {
    Box(
        modifier = modifier,
        contentAlignment = contentAlignment
    ) {
        Text(
            text = "Hello $name!",
            fontSize = 20.sp
        )
    }
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    DominionEnergyPolepadAITheme {
        Greeting("Android")
    }
}
