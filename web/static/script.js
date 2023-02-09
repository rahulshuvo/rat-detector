function showImage(url) {
  document.getElementById("image").src = url;
  document.getElementById("image-container").style.display = "block";
  document.getElementById('body').style.overflow = "hidden";
  document.getElementById('body').style.height= "100vh";

}

function hideImage() {
  document.getElementById("image-container").style.display = "none";
  document.getElementById('body').style.overflow = "unset";
  document.getElementById('body').style.height= "unset";
}


