package com.dominionenergy.polepadai.model

import android.location.Location
import androidx.annotation.DrawableRes
import androidx.annotation.StringRes

/*
 * poleID: String of the pole's ID.
 * imageResourceIds: Array of integers for the image(s) of the poles.
 * poleType: "wooden" or "composite"
 * surroundings: "transformer", "insulator", "streetlight"
 * vegetation: 0, 1, or 2 correlated to none, low, or high.
 * location: Location object representing where the pole is.
 */
data class PoleData(
    val poleID: String = "",
    val imageResourceIds: Array<Int>,
    val poleType: String = "",
    val surroundings: Array<String>,
    val vegetation: Int,
    val location : Location
)