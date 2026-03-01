package com.dominionenergy.polepadai

import android.R.attr.onClick
import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.dominionenergy.polepadai.ui.theme.DominionBlue


@Composable
fun FormVerificationScreenUI() {
    // MAKE THIS DYNAMIC LATER
    var poleID by remember { mutableStateOf("") }
    var poleType by remember { mutableStateOf("") }
    var surroundings by remember {
        mutableStateOf(
            listOf(
                false,
                false,
                false
            )
        )
    }// 3 bools we are making them false to start wit
    var vegetation by remember { mutableStateOf(0) } // it'll be 0 or 1 or 2

    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState) // scroll
            .padding(16.dp),
        horizontalAlignment = Alignment.Start
    ) {
        Spacer(modifier = Modifier.height(50.dp)) // space above image to not touch top
        // placeholder
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(400.dp), // made bigger --> make this scrollable
            contentAlignment = Alignment.Center
        ) {
            Image(
                painter = painterResource(R.drawable.dominionlogo), //placeholder
                contentDescription = "Pole Image Placeholder",
                modifier = Modifier.fillMaxSize()
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Pole ID
        OutlinedTextField( // MAKE DYNAMIC
            value = poleID,
            onValueChange = { poleID = it },
            label = { Text("Pole ID") },
            modifier = Modifier
                .fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        //Pole Type
        OutlinedTextField(
            value = poleType,
            onValueChange = { poleType = it },
            label = { Text("Pole Type") },
            modifier = Modifier
                .fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Surroundings checkboxes
        Text("Surroundings:", fontSize = 16.sp)
        val surroundingsLabels = listOf("Transformer", "Insulator", "Streetlight")
        surroundingsLabels.forEachIndexed { index, label ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(
                    checked = surroundings[index],
                    onCheckedChange = { isChecked ->
                        surroundings = surroundings.toMutableList().also { it[index] = isChecked }
                    }
                )
                Text(label)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Vegetation radio buttons
        Text("Vegetation:", fontSize = 16.sp)
        val vegetationOptions = listOf("None", "Low", "High")
        vegetationOptions.forEachIndexed { index, label ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(
                    selected = vegetation == index,
                    onClick = { vegetation = index }
                )
                Text(label)
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        // submitting --> connect to uploads to get in the gallery
        Button(
            onClick = { /* BE ABLE TO SUBMIT */ },
            modifier = Modifier.align(Alignment.CenterHorizontally), // horizontal centered
            colors = ButtonDefaults.buttonColors(
                containerColor = DominionBlue,
                contentColor = Color.White
            )
        ) {
            Text("Submit")
        }
    }
}