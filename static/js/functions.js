/*
	When the bandcamp link is pressed, stop all propagation so AmplitudeJS doesn't
	play the song.
*/
let bandcampLinks = document.getElementsByClassName('bandcamp-link');

for( var i = 0; i < bandcampLinks.length; i++ ){
	bandcampLinks[i].addEventListener('click', function(e){
		e.stopPropagation();
	});
}
function capture() {
	fetch('/capture')
		.then(response => response.text())
		.then(message => alert(message));
}

let songElements = document.getElementsByClassName('song');

for( var i = 0; i < songElements.length; i++ ){
	/*
		Ensure that on mouseover, CSS styles don't get messed up for active songs.
	*/
	songElements[i].addEventListener('mouseover', function(){
		this.style.backgroundColor = '#00A0FF';

		this.querySelectorAll('.song-meta-data .song-title')[0].style.color = '#FFFFFF';
		this.querySelectorAll('.song-meta-data .song-artist')[0].style.color = '#FFFFFF';

		if( !this.classList.contains('amplitude-active-song-container') ){
			this.querySelectorAll('.play-button-container')[0].style.display = 'block';
		}

		this.querySelectorAll('img.bandcamp-grey')[0].style.display = 'none';
		this.querySelectorAll('img.bandcamp-white')[0].style.display = 'block';
		this.querySelectorAll('.song-duration')[0].style.color = '#FFFFFF';
	});

	/*
		Ensure that on mouseout, CSS styles don't get messed up for active songs.
	*/
	songElements[i].addEventListener('mouseout', function(){
		this.style.backgroundColor = '#FFFFFF';
		this.querySelectorAll('.song-meta-data .song-title')[0].style.color = '#272726';
		this.querySelectorAll('.song-meta-data .song-artist')[0].style.color = '#607D8B';
		this.querySelectorAll('.play-button-container')[0].style.display = 'none';
		this.querySelectorAll('img.bandcamp-grey')[0].style.display = 'block';
		this.querySelectorAll('img.bandcamp-white')[0].style.display = 'none';
		this.querySelectorAll('.song-duration')[0].style.color = '#607D8B';
	});

	/*
		Show and hide the play button container on the song when the song is clicked.
	*/
	songElements[i].addEventListener('click', function(){
		this.querySelectorAll('.play-button-container')[0].style.display = 'none';
	});
}

/*
	Initializes AmplitudeJS
*/
Amplitude.init({
	continue_next: false,
	callbacks: {
		song_change: function(){
			alert('here');
		}
	},
	"songs": [
		{
			"name": "Generated Music",
			"artist": "MUZIC MAESTEROS",
			"album": "We Are to Answer",
			"url": "http://127.0.0.1:5000/audio/audio1.wav",
			"cover_art_url": "{{../album-art/we-are-to-answer.jpg}}"
		},
		{
			"name": "The Gun",
			"artist": "MUZIC MAESTEROS",
			"album": "Ask The Dust",
			"url": "{{ audio }}",
			"cover_art_url": "../album-art/ask-the-dust.jpg"
		},
		{
			"name": "Anvil",
			"artist": "MUZIC MAESTEROS",
			"album": "Anvil",
			"url": "http://127.0.0.1:5000/audio/audio2.wav",
			"cover_art_url": "../album-art/anvil.jpg"
		},
		{
			"name": "I Came Running",
			"artist": "Ancient Astronauts",
			"album": "We Are to Answer",
			"url": "http://127.0.0.1:5000/audio/audio2.wav",
			"cover_art_url": "../album-art/we-are-to-answer.jpg"
		},
		{
			"name": "First Snow",
			"artist": "Emancipator",
			"album": "Soon It Will Be Cold Enough",
			"url": "http://127.0.0.1:5000/audio/audio2.wav",
			"cover_art_url": "../album-art/soon-it-will-be-cold-enough.jpg"
		},
		{
			"name": "Terrain",
			"artist": "pg.lost",
			"album": "Key",
			"url": "http://127.0.0.1:5000/audio/audio2.wav",
			"cover_art_url": "../album-art/key.jpg"
		},
		{
			"name": "Vorel",
			"artist": "Russian Circles",
			"album": "Guidance",
			"url": "http://127.0.0.1:5000/audio/audio2.wav",
			"cover_art_url": "../album-art/guidance.jpg"
		}
	]
});
