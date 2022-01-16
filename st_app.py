import math

import streamlit as st

import classifier


def main():
    st.set_page_config("SponsorBlock AutoMod", "🤖")
    st.title("SponsorBlock AutoMod")
    submission_id = st.text_input("Submission ID (Sponsor / Self Promo Only)")
    if not submission_id:
        return
    automod_ret = classifier.classify_uuid(submission_id)
    if "error" in automod_ret:
        return st.error(f"Error Occured. [Error : {automod_ret['error']}]")
    start_time = math.floor(automod_ret["start_time"])
    end_time = math.floor(automod_ret["end_time"])
    st.markdown(
        """<style>.videoWrapper {
		position: relative;
		padding-bottom: 56.25%; /* 16:9 */
		padding-top: 25px;
		height: 0;
	}
	.videoWrapper iframe {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
	}</style>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="videoWrapper"><iframe src="https://www.youtube-nocookie.com/embed/{automod_ret["video_id"]}?start={start_time}&end={end_time}"></iframe></div>',
        unsafe_allow_html=True,
    )

    if automod_ret["is_sponsored"]:
        st.success('This portion of the video was sponsored.')
    else:
        st.error('This portion of the video was not sponsored.')
    st.info(f"Transcript : {automod_ret['text']}")


if __name__ == "__main__":
    main()
